// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/program.hpp"
#include "program_node.h"
#include <vector>
#include <map>
#include <algorithm>
#include <set>
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"

namespace cldnn {
// helper method for calc_processing order
void program::nodes_ordering::calc_processing_order_visit(program_node* node) {
    if (node->is_marked())
        return;
    for (auto user : node->users) {
        calc_processing_order_visit(user);
    }
    node->mark();
    _processing_order.push_front(node);
    processing_order_iterators[node] = _processing_order.begin();
    return;
}

// DFS to sort nodes topologically
// any topological sort of nodes is required for further optimizations
void program::nodes_ordering::calc_processing_order(program& p) {
    _processing_order.clear();
    for (auto input : p.get_inputs()) {
        calc_processing_order_visit(input);
    }
    for (auto& node : _processing_order) {
        node->unmark();
    }
    return;
}

/*
    recalculate processing_order
    algorithm based on: CLRS 24.5 (critical path in DAG)
    modifications: adjust for multiple inputs
    input: any topological order in processing order
    output: BFS topological order.
    */
void program::nodes_ordering::calculate_BFS_processing_order() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("calculate_BFS_processing_order");
    std::map<program_node*, int> distances;
    for (auto itr : _processing_order) {
        distances[itr] = -1;
    }
    int max_distance = 0;
    for (auto itr : _processing_order) {
        // Init
        if (distances[itr] == -1) {  // this must be an input
            distances[itr] = 0;      // initialize input
        }
        // RELAX
        for (auto& user : itr->get_users()) {
            distances[user] = std::max(distances[user], distances[itr] + 1);
            max_distance = std::max(max_distance, distances[user]);
        }
    }

    // bucket sort nodes based on their max distance from input
    std::vector<std::vector<program_node*>> dist_lists;
    dist_lists.resize(max_distance + 1);
    for (auto itr : _processing_order) {
        dist_lists[distances[itr]].push_back(itr);
    }

    // replace the old processing order by the new one, still topological.
    _processing_order.clear();
    for (auto& dist : dist_lists) {
        for (auto& node : dist) {
            _processing_order.push_back(node);
            processing_order_iterators[node] = _processing_order.end();
            processing_order_iterators[node]--;
        }
    }
    return;
}


static void dfs(program_node* node, std::set<program_node*> &visited, std::vector<program_node *> &order) {
    std::cout << "DFS: Try visit node: " << node->id() << std::endl;
    if (visited.find(node) != visited.end())
        return;
    
    for (auto dep_pair: node->get_dependencies()) {
        std::cout << "DFS:   check dependency: " << dep_pair.first->id() << std::endl;
        // if any of node's dependency is not visited, we cannot process it at this moment.
        auto dep = dep_pair.first;
        if (dep->is_type<data>() || dep->is_type<mutable_data>()) {
            visited.insert(dep);
            order.push_back(dep);
        }

        if (visited.find(dep_pair.first) == visited.end()) {
            std::cout << "DFS:   not ready..." << std::endl;
            return;
        }
    }

    visited.insert(node);
    order.push_back(node);

    for (auto user : node->get_users()) {
        dfs(user, visited, order);
    }
}


 void program::nodes_ordering::calculate_DFS_processing_order() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("calculate_DFS_processing_order");
    std::set<program_node*> visited;
    std::vector<program_node *> order;
    for (auto node : _processing_order)
        dfs(node, visited, order);
    _processing_order.clear();
    for (auto node : order) {
        _processing_order.push_back(node);
        std::cout << __func__ << ": " << node->id() << std::endl;
        processing_order_iterators[node] = _processing_order.end();
        processing_order_iterators[node]--;
    }
    return;
}

// verifies if a given node will be processed before all its dependent nodes
bool program::nodes_ordering::is_correct(program_node* node) {
    for (auto& dep : node->get_dependencies()) {
        if (get_processing_number(node) < get_processing_number(dep.first)) {
            return false;
        }
    }
    return true;
}
}  // namespace cldnn
