/*
 * This program is part of the paper "Multi-Objective A* Algorithm for the
 * Multimodal Multi-Objective Path Planning Optimization".
 *
 * Copyright (c) 2021 Bo Jin <jinbostar@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "nlohmann/json.hpp"
#include "pareto/front.h"
#include <fstream>
#include <memory>
#include <set>
#include <unordered_set>
using namespace nlohmann;
using namespace pareto;

using namespace std;

//#define PRINT_LOG

#define within(x, y, X, Y) ((x) >= (0) && (x) < (X) && (y) >= (0) && (y) < (Y))

template <typename T> using Matrix = vector<vector<T>>;
using Coord = pair<int /*x*/, int /*y*/>;
using Path = vector<Coord>;
using Group = pair<vector<double> /*obj_cost*/, vector<Path> /*paths*/>;
const int steps[4][2] = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};

class Problem {
public:
  int start_x;
  int start_y;
  int goal_x;
  int goal_y;
  int dim_x;
  int dim_y;
  int dim_k;
  int dim_c;
  Matrix<bool> passable;
  vector<Coord> key_list;
  Matrix<int> key_matrix;
  Matrix<vector<double>> cost_matrix;
  vector<pair<int, double>> factors;

  pair<int, int> size() const { return size(passable); }

  pair<int, int> size(const Matrix<bool> &retained) const {
    int num_areas = 0;
    int num_links = 0;
    for (int x = 0; x < dim_x; x++) {
      for (int y = 0; y < dim_y; y++) {
        if (retained[x][y]) {
          num_areas++;
          num_links += (x > 0 && retained[x - 1][y] ? 1 : 0) +
                       (x < dim_x - 1 && retained[x + 1][y] ? 1 : 0) +
                       (y > 0 && retained[x][y - 1] ? 1 : 0) +
                       (y < dim_y - 1 && retained[x][y + 1] ? 1 : 0);
        }
      }
    }
    return {num_areas, num_links};
  }

  Matrix<bool> reduce() const {
    Matrix<int> depth(dim_x, vector<int>(dim_y, 0));
    Matrix<int> low(dim_x, vector<int>(dim_y, 0));
    Matrix<bool> flag(dim_x, vector<bool>(dim_y, false));
    Matrix<vector<Coord>> children(dim_x, vector<vector<Coord>>(dim_y));
    Matrix<bool> retained(dim_x, vector<bool>(dim_y, false));

    function<void(int, int, int)> build_tree;
    build_tree = [&](int x, int y, int d) {
      depth[x][y] = d;
      low[x][y] = d;
      flag[x][y] = x == start_x && y == start_y || x == goal_x && y == goal_y ||
                   key_matrix[x][y] != -1;

      for (auto [step_x, step_y] : steps) {
        int child_x = x + step_x;
        int child_y = y + step_y;
        if (within(child_x, child_y, dim_x, dim_y) &&
            passable[child_x][child_y]) {
          if (depth[child_x][child_y] == 0) {
            children[x][y].emplace_back(child_x, child_y);
            build_tree(child_x, child_y, d + 1);
            if (low[x][y] > low[child_x][child_y]) {
              low[x][y] = low[child_x][child_y];
            }
            if (flag[child_x][child_y]) {
              flag[x][y] = true;
            }
          } else if (low[x][y] > depth[child_x][child_y]) {
            low[x][y] = depth[child_x][child_y];
          }
        }
      }
    };

    function<void(int, int)> trim_tree;
    trim_tree = [&](int x, int y) {
      retained[x][y] = true;
      for (auto [next_x, next_y] : children[x][y]) {
        if (low[next_x][next_y] < depth[x][y] || flag[next_x][next_y]) {
          trim_tree(next_x, next_y);
        }
      }
    };

    if (passable[start_x][start_y]) {
      build_tree(start_x, start_y, 1);
      trim_tree(start_x, start_y);
    }

    return retained;
  }
};

enum Objective { unknown, min_path_len, min_num_red, min_num_cross, min_f };

Objective parse_objective(const string &obj) {
  if (obj == "min_path_len") {
    return min_path_len;
  } else if (obj == "min_num_red") {
    return min_num_red;
  } else if (obj == "min_num_cross") {
    return min_num_cross;
  } else if (obj == "min_f") {
    return min_f;
  } else {
    return unknown;
  }
}

vector<Objective> get_default_objs(int prob_id) {
  if (prob_id == 1) {
    return {min_path_len, min_num_red};
  } else if (prob_id <= 5) {
    return {min_path_len, min_num_red, min_num_cross};
  } else {
    return {min_path_len, min_f};
  }
}

Problem parse_problem(const json &data, const vector<Objective> &objs) {
  int start_x = data["START_x"].get<int>() - 1;
  int start_y = data["START_y"].get<int>() - 1;
  int goal_x = data["GOAL_x"].get<int>() - 1;
  int goal_y = data["GOAL_y"].get<int>() - 1;

  auto MAP = data["Map"].get<Matrix<int>>();
  int dim_x = (int)MAP[0].size();
  int dim_y = (int)MAP.size();

  Matrix<bool> passable(dim_x, vector<bool>(dim_y, false));
  for (int x = 0; x < dim_x; x++) {
    for (int y = 0; y < dim_y; y++) {
      if (MAP[y][x] == 0) {
        passable[x][y] = true;
      }
    }
  }

  vector<Coord> key_list;
  Matrix<int> key_matrix(dim_x, vector<int>(dim_y, -1));
  int dim_k = 0;
  if (data.contains("Yellow_areas")) {
    auto YELLOW_AREAS = data["Yellow_areas"].get<set<pair<int, int>>>();
    YELLOW_AREAS.erase({start_x + 1, start_y + 1});
    YELLOW_AREAS.erase({goal_x + 1, goal_y + 1});
    for (auto [X, Y] : YELLOW_AREAS) {
      key_list.emplace_back(X - 1, Y - 1);
      key_matrix[X - 1][Y - 1] = dim_k++;
    }
  }

  Matrix<vector<double>> cost_matrix(dim_x, vector<vector<double>>(dim_y));
  int dim_c = 0;
  vector<pair<int, double>> factors;
  for (Objective obj : objs) {
    if (obj == min_path_len) {
      for (int x = 0; x < dim_x; x++) {
        for (int y = 0; y < dim_y; y++) {
          if (passable[x][y]) {
            cost_matrix[x][y].push_back(1);
          }
        }
      }
      dim_c++;
    } else if (obj == min_num_red) {
      auto RED_AREAS = data["Red_areas"].get<set<pair<int, int>>>();
      for (int x = 0; x < dim_x; x++) {
        for (int y = 0; y < dim_y; y++) {
          if (passable[x][y]) {
            if (RED_AREAS.find({x + 1, y + 1}) != RED_AREAS.end()) {
              cost_matrix[x][y].push_back(1);
            } else {
              cost_matrix[x][y].push_back(0);
            }
          }
        }
      }
      dim_c++;
    } else if (obj == min_num_cross) {
      for (int x = 0; x < dim_x; x++) {
        for (int y = 0; y < dim_y; y++) {
          if (passable[x][y]) {
            int degree = (x > 0 && passable[x - 1][y] ? 1 : 0) +
                         (x < dim_x - 1 && passable[x + 1][y] ? 1 : 0) +
                         (y > 0 && passable[x][y - 1] ? 1 : 0) +
                         (y < dim_y - 1 && passable[x][y + 1] ? 1 : 0);
            if (degree >= 3) {
              cost_matrix[x][y].push_back(1);
            } else {
              cost_matrix[x][y].push_back(0);
            }
          }
        }
      }
      dim_c++;
    } else if (obj == min_f) {
      auto F = data["F"].get<Matrix<double>>();
      int dim_f = (int)F[0].size() - 2;
      map<pair<int, int>, vector<double>> F_MAP;
      for (const vector<double> &row : F) {
        F_MAP[{lround(row[0]), lround(row[1])}] =
            vector<double>(row.begin() + 2, row.end());
      }
      for (int x = 0; x < dim_x; x++) {
        for (int y = 0; y < dim_y; y++) {
          if (passable[x][y]) {
            if (auto it = F_MAP.find({x + 1, y + 1}); it != F_MAP.end()) {
              for (double f : it->second) {
                cost_matrix[x][y].push_back(round(f * 10));
              }
            } else {
              for (int f = 0; f < dim_f; f++) {
                cost_matrix[x][y].push_back(0);
              }
            }
          }
        }
      }
      for (int f = 0; f < dim_f; f++) {
        factors.emplace_back(dim_c, 10);
        dim_c++;
      }
    }
  }

  return {start_x,
          start_y,
          goal_x,
          goal_y,
          dim_x,
          dim_y,
          dim_k,
          dim_c,
          move(passable),
          move(key_list),
          move(key_matrix),
          move(cost_matrix),
          move(factors)};
}

Problem get_default_problem(int prob_id) {
  json data;
  ifstream ifs("data/Problem_" + to_string(prob_id) + ".json");
  ifs >> data;
  ifs.close();

  vector<Objective> objs = get_default_objs(prob_id);

  return parse_problem(data, objs);
}

Problem get_problem(const string &file, const vector<string> &args) {
  json data;
  ifstream ifs(file);
  ifs >> data;
  ifs.close();

  vector<Objective> objs;
  for (const string &arg : args) {
    Objective obj = parse_objective(arg);
    if (obj != unknown) {
      objs.push_back(obj);
    }
  }

  return parse_problem(data, objs);
}

class BaseSolver {
public:
  virtual vector<Group> solve(const Problem &prob) const = 0;
};

template <size_t M = 0> class Solver : public BaseSolver {
private:
  using Cost = point<double, M>;

  class CostComparator {
  public:
    constexpr bool operator()(const Cost &a, const Cost &b) const {
      return lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
    }
  };

  struct Triplet {
    int node;
    int status;
    Cost cost;
  };

  class TripletComparator {
  public:
    constexpr bool operator()(const Triplet &a, const Triplet &b) const {
      return a.node < b.node ||
             a.node == b.node &&
                 (a.status < b.status ||
                  a.status == b.status &&
                      lexicographical_compare(a.cost.begin(), a.cost.end(),
                                              b.cost.begin(), b.cost.end()));
    }
  };

  struct Previous {
    int prev_node;
    int prev_status;
    Cost prev_cost;
    int edge_id;
  };

  struct Edge {
    int prev_node;
    int next_node;
    Cost edge_cost;
    Path edge_pass;
  };

  struct Extra {
    Cost est;
    vector<Previous> previous;
  };

  using Front = front<double, M, Extra>;
  using Queue = map<Cost, set<Triplet, TripletComparator>, CostComparator>;

public:
  vector<Group> solve(const Problem &prob) const override {
    const auto &[start_x, start_y, goal_x, goal_y, dim_x, dim_y, dim_k, dim_c,
                 passable, key_list, key_matrix, cost_matrix, factors] = prob;

#ifdef PRINT_LOG
    auto time_point_0 = chrono::high_resolution_clock::now();
#endif

    /*
     * Map Reduction
     */
    Matrix<bool> retained = prob.reduce();

    bool solvable = retained[start_x][start_y] && retained[goal_x][goal_y];
    for (int k = 0; solvable && k < dim_k; k++) {
      auto [key_x, key_y] = key_list[k];
      if (!retained[key_x][key_y]) {
        solvable = false;
      }
    }

    if (!solvable) {
#ifdef PRINT_LOG
      cout << "\t"
           << "Unsolvable!" << endl;
#endif
      return {};
    }

    Matrix<int> degree_matrix(dim_x, vector<int>(dim_y, 0));
    for (int x = 0; x < dim_x; x++) {
      for (int y = 0; y < dim_y; y++) {
        if (retained[x][y]) {
          degree_matrix[x][y] = (x > 0 && retained[x - 1][y] ? 1 : 0) +
                                (x < dim_x - 1 && retained[x + 1][y] ? 1 : 0) +
                                (y > 0 && retained[x][y - 1] ? 1 : 0) +
                                (y < dim_y - 1 && retained[x][y + 1] ? 1 : 0);
        }
      }
    }

#ifdef PRINT_LOG
    auto [num_areas, num_links] = prob.size(retained);
    auto time_point_1 = chrono::high_resolution_clock::now();
    cout << "\t"
         << "Map Reduction ["
         << chrono::duration<double>(time_point_1 - time_point_0).count() * 1000
         << "ms]: " << num_areas << " Nodes, " << num_links / 2 << " Edges"
         << endl;
#endif

    /*
     * Graph Model
     */
    int start_node = -1;
    int goal_node = -1;
    vector<int> key_nodes(dim_k);
    Matrix<int> node_matrix(dim_x, vector<int>(dim_y, -1));
    int dim_n = 0;
    vector<Coord> coords;
    vector<int> keys;
    vector<Cost> costs;

    for (int x = 0; x < dim_x; x++) {
      for (int y = 0; y < dim_y; y++) {
        if (x == start_x && y == start_y || x == goal_x && y == goal_y ||
            key_matrix[x][y] != -1 || degree_matrix[x][y] >= 3) {
          if (x == start_x && y == start_y) {
            start_node = dim_n;
          }
          if (x == goal_x && y == goal_y) {
            goal_node = dim_n;
          }
          if (key_matrix[x][y] != -1) {
            key_nodes[key_matrix[x][y]] = dim_n;
          }

          node_matrix[x][y] = dim_n;
          coords.emplace_back(x, y);
          keys.push_back(key_matrix[x][y]);
          costs.emplace_back(cost_matrix[x][y].begin(),
                             cost_matrix[x][y].end());

          dim_n++;
        }
      }
    }

    vector<Edge> edges;
    vector<vector<int>> out_edges(dim_n);
    vector<vector<int>> in_edges(dim_n);
    for (int prev_node = 0; prev_node < dim_n; prev_node++) {
      auto [x, y] = coords[prev_node];
      for (const auto &[step_x, step_y] : steps) {
        if (within(x + step_x, y + step_y, dim_x, dim_y) &&
            retained[x + step_x][y + step_y]) {
          Cost edge_cost(dim_c, 0);
          Path edge_pass;
          int prev_x = x;
          int prev_y = y;
          int curr_x = x + step_x;
          int curr_y = y + step_y;
          while (node_matrix[curr_x][curr_y] == -1) {
            for (int c = 0; c < dim_c; c++) {
              edge_cost[c] += cost_matrix[curr_x][curr_y][c];
            }
            edge_pass.emplace_back(curr_x, curr_y);

            for (const auto &[next_step_x, next_step_y] : steps) {
              if (within(curr_x + next_step_x, curr_y + next_step_y, dim_x,
                         dim_y) &&
                  retained[curr_x + next_step_x][curr_y + next_step_y] &&
                  !(curr_x + next_step_x == prev_x &&
                    curr_y + next_step_y == prev_y)) {
                prev_x = curr_x;
                prev_y = curr_y;
                curr_x += next_step_x;
                curr_y += next_step_y;
                break;
              }
            }
          }

          int next_node = node_matrix[curr_x][curr_y];
          int edge_id = edges.size();
          out_edges[prev_node].push_back(edge_id);
          in_edges[next_node].push_back(edge_id);
          edges.push_back(
              {prev_node, next_node, move(edge_cost), move(edge_pass)});
        }
      }
    }

#ifdef PRINT_LOG
    auto time_point_2 = chrono::high_resolution_clock::now();
    cout << "\t"
         << "Graph Model ["
         << chrono::duration<double>(time_point_2 - time_point_1).count() * 1000
         << "ms]: " << dim_n << " Nodes, " << edges.size() / 2 << " Edges"
         << endl;
#endif

    /*
     * Ideal Costs
     */
    Matrix<Cost> ideal_to_keys(dim_n, vector<Cost>(dim_k, Cost(dim_c, 1e9)));
    for (int c = 0; c < dim_c; c++) {
      for (int k = 0; k < dim_k; k++) {
        ideal_to_keys[key_nodes[k]][k][c] = 0;
        set<pair<double, int>> pq;
        pq.emplace(0, key_nodes[k]);
        while (!pq.empty()) {
          auto [dist, node] = *pq.begin();
          pq.erase(pq.begin());
          for (int edge_id : in_edges[node]) {
            int prev_node = edges[edge_id].prev_node;
            double new_prev_dist = edges[edge_id].edge_cost[c] + dist;
            if (node != key_nodes[k]) {
              new_prev_dist += costs[node][c];
            }
            if (ideal_to_keys[prev_node][k][c] == 1e9) {
              ideal_to_keys[prev_node][k][c] = new_prev_dist;
              pq.emplace(new_prev_dist, prev_node);
            } else if (new_prev_dist < ideal_to_keys[prev_node][k][c]) {
              pq.erase({ideal_to_keys[prev_node][k][c], prev_node});
              ideal_to_keys[prev_node][k][c] = new_prev_dist;
              pq.emplace(new_prev_dist, prev_node);
            }
          }
        }
      }
    }

    vector<Cost> ideal_to_goal(dim_n, Cost(dim_c, 1e9));
    for (int c = 0; c < dim_c; c++) {
      ideal_to_goal[goal_node][c] = 0;
      set<pair<double, int>> pq;
      pq.emplace(0, goal_node);
      while (!pq.empty()) {
        auto [dist, node] = *pq.begin();
        pq.erase(pq.begin());
        for (int edge_id : in_edges[node]) {
          int prev_node = edges[edge_id].prev_node;
          double new_prev_dist = edges[edge_id].edge_cost[c] + dist;
          if (node != goal_node) {
            new_prev_dist += costs[node][c];
          }
          if (ideal_to_goal[prev_node][c] == 1e9) {
            ideal_to_goal[prev_node][c] = new_prev_dist;
            pq.emplace(new_prev_dist, prev_node);
          } else if (new_prev_dist < ideal_to_goal[prev_node][c]) {
            pq.erase({ideal_to_goal[prev_node][c], prev_node});
            ideal_to_goal[prev_node][c] = new_prev_dist;
            pq.emplace(new_prev_dist, prev_node);
          }
        }
      }
    }

    /*
     * Best-First Search
     */
    auto heuristic = [&, dim_k = dim_k, dim_c = dim_c,
                      heu_mem = vector<unordered_map<int, Cost>>(dim_n),
                      mst_mem = unordered_map<int, Cost>()](
                         int node, int status) mutable {
      if (auto it = heu_mem[node].find(status); it != heu_mem[node].end()) {
        return it->second;
      }

      if (status == (1 << dim_k) - 1) {
        if (node == goal_node) {
          return heu_mem[node][status] = Cost(dim_c, 0);
        } else {
          return heu_mem[node][status] = ideal_to_goal[node] + costs[goal_node];
        }
      }

      Cost lb = costs[goal_node];
      vector<int> key_ids;
      for (int k = 0; k < dim_k; k++) {
        if ((status & (1 << k)) == 0) {
          key_ids.push_back(k);
          lb += costs[key_nodes[k]];
        }
      }

      if (key_ids.size() == 1) {
        lb += ideal_to_keys[node][key_ids[0]];
        lb += ideal_to_goal[key_nodes[key_ids[0]]];
      } else if (key_ids.size() == 2) {
        for (int c = 0; c < dim_c; c++) {
          double dist_a = ideal_to_keys[node][key_ids[0]][c] +
                          ideal_to_keys[key_nodes[key_ids[0]]][key_ids[1]][c] +
                          ideal_to_goal[key_nodes[key_ids[1]]][c];
          double dist_b = ideal_to_keys[node][key_ids[1]][c] +
                          ideal_to_keys[key_nodes[key_ids[1]]][key_ids[0]][c] +
                          ideal_to_goal[key_nodes[key_ids[0]]][c];
          lb[c] += dist_a < dist_b ? dist_a : dist_b;
        }
      } else {
        for (int c = 0; c < dim_c; c++) {
          double min_from_curr = 1e9;
          double min_to_goal = 1e9;
          for (int k : key_ids) {
            if (min_from_curr > ideal_to_keys[node][k][c]) {
              min_from_curr = ideal_to_keys[node][k][c];
            }
            if (min_to_goal > ideal_to_goal[key_nodes[k]][c]) {
              min_to_goal = ideal_to_goal[key_nodes[k]][c];
            }
          }
          lb[c] += min_from_curr + min_to_goal;
        }

        if (auto it = mst_mem.find(status); it != mst_mem.end()) {
          lb += it->second;
        } else {
          Cost mst_cost(dim_c, 0);
          for (int c = 0; c < dim_c; c++) {
            vector<tuple<double, int, int>> links;
            for (int i = 0; i < key_ids.size(); i++) {
              for (int j = i + 1; j < key_ids.size(); j++) {
                double dist_a =
                    ideal_to_keys[key_nodes[key_ids[i]]][key_ids[j]][c];
                double dist_b =
                    ideal_to_keys[key_nodes[key_ids[j]]][key_ids[i]][c];
                links.emplace_back(dist_a < dist_b ? dist_a : dist_b, i, j);
              }
            }
            sort(links.begin(), links.end());

            vector<int> leaders(key_ids.size());
            for (int i = 0; i < key_ids.size(); i++) {
              leaders[i] = i;
            }

            double total_weight = 0;
            int num_links_added = 0;
            for (auto [w, i, j] : links) {
              if (num_links_added == key_ids.size() - 1) {
                break;
              }

              int li = leaders[i];
              while (li != leaders[li]) {
                li = leaders[li];
              }
              for (int pi = i; leaders[pi] != li;) {
                int old = leaders[pi];
                leaders[pi] = li;
                pi = old;
              }

              int lj = leaders[j];
              while (lj != leaders[lj]) {
                lj = leaders[lj];
              }
              for (int pj = j; leaders[pj] != lj;) {
                int old = leaders[pj];
                leaders[pj] = lj;
                pj = old;
              }

              if (li != lj) {
                leaders[li] = lj;
                total_weight += w;
                num_links_added++;
              }
            }

            mst_cost[c] = total_weight;
          }

          lb += (mst_mem[status] = mst_cost);
        }
      }

      return heu_mem[node][status] = lb;
    };

    const Cost &start_cost = costs[start_node];

    Cost start_est = start_cost + heuristic(start_node, 0);

    vector<unordered_map<int, Front>> tentative(dim_n);
    tentative[start_node][0][start_cost] = {start_est, {}};

    const Front &goal = tentative[goal_node][(1 << dim_k) - 1];

    Queue open;
    open[start_est] = {{start_node, 0, costs[start_node]}};

    int iteration = 0;
    while (!open.empty()) {
      iteration++;
      Cost est = open.begin()->first;
      auto [node, status, cost] = *open.begin()->second.begin();

      if (auto it = open.begin(); it->second.size() == 1) {
        open.erase(open.begin());
      } else {
        it->second.erase(it->second.begin());
      }

      if (node == goal_node && status == (1 << dim_k) - 1) {
        continue;
      }

      if (goal.dominates(est)) {
        tentative[node][status].erase(cost);
        continue;
      }

      for (int edge_id : out_edges[node]) {
        int next_node = edges[edge_id].next_node;

        int next_status = status;
        if (keys[next_node] != -1) {
          next_status |= (1 << keys[next_node]);
        }

        Cost next_cost = cost;
        next_cost += edges[edge_id].edge_cost;
        next_cost += costs[next_node];

        Front &next_tent = tentative[next_node][next_status];

        if (next_tent.dominates(next_cost)) {
          continue;
        }

        if (auto it = next_tent.find(next_cost); it != next_tent.end()) {
          it->second.previous.push_back({node, status, cost, edge_id});
          continue;
        }

        if (auto it_dom = next_tent.find_dominated(next_cost);
            it_dom != next_tent.end()) {
          vector<Cost> dominated;
          while (it_dom != next_tent.end()) {
            const Cost &dom_cost = it_dom->first;
            const Cost &dom_est = it_dom->second.est;
            auto it_open = open.find(dom_est);
            auto it_set =
                it_open->second.find({next_node, next_status, dom_cost});
            if (it_open->second.size() == 1) {
              open.erase(it_open);
            } else {
              it_open->second.erase(it_set);
            }
            dominated.push_back(dom_cost);
            it_dom++;
          }
          for (const Cost &dom_cost : dominated) {
            next_tent.erase(dom_cost);
          }
        }

        Cost next_est = next_cost + heuristic(next_node, next_status);
        if (goal.dominates(next_est)) {
          continue;
        }

        open[next_est].insert({next_node, next_status, next_cost});

        next_tent[next_cost] = {next_est, {{node, status, cost, edge_id}}};
      }
    }

#ifdef PRINT_LOG
    auto time_point_3 = chrono::high_resolution_clock::now();

    cout << "\t"
         << "Best-First Search ["
         << chrono::duration<double>(time_point_3 - time_point_2).count() * 1000
         << "ms]: " << iteration << " Iterations" << endl;
#endif

    /*
     * Path Reconstruction
     */
    int num_paths = 0;
    vector<Group> groups;
    for (const auto &[goal_cost, goal_extra] : goal) {
      vector<Path> paths;

      function<void(int, int, const Cost &)> backtrack;
      backtrack = [&, rev_path = Path(),
                   visited = vector<unordered_set<int>>(dim_n)](
                      int node, int status, const Cost &cost) mutable {
        rev_path.push_back(coords[node]);
        visited[node].insert(status);

        const vector<Previous> &previous =
            tentative[node][status][cost].previous;
        if (node == start_node && status == 0) {
          paths.emplace_back(rev_path.rbegin(), rev_path.rend());
        } else {
          int rev_path_size = (int)rev_path.size();
          for (const auto &[prev_node, prev_status, prev_cost, edge_id] :
               previous) {
            if (visited[prev_node].find(prev_status) ==
                visited[prev_node].end()) {
              const Path &edge_pass = edges[edge_id].edge_pass;
              rev_path.insert(rev_path.end(), edge_pass.rbegin(),
                              edge_pass.rend());
              backtrack(prev_node, prev_status, prev_cost);
              rev_path.resize(rev_path_size);
            }
          }
        }

        rev_path.pop_back();
        visited[node].erase(status);
      };

      backtrack(goal_node, (1 << dim_k) - 1, goal_cost);

      vector<double> obj_cost(goal_cost.begin(), goal_cost.end());
      for (auto [index, factor] : factors) {
        obj_cost[index] /= factor;
      }

      num_paths += (int)paths.size();
      groups.emplace_back(obj_cost, move(paths));
    }

#ifdef PRINT_LOG
    auto time_point_4 = chrono::high_resolution_clock::now();

    cout << "\t"
         << "Path Construction ["
         << chrono::duration<double>(time_point_4 - time_point_3).count() * 1000
         << "ms]: " << groups.size() << " Points, " << num_paths << " Paths"
         << endl;

    cout << "\t"
         << "Total Time ["
         << chrono::duration<double>(time_point_4 - time_point_0).count() * 1000
         << "ms]" << endl;
#endif

    return groups;
  }
};

shared_ptr<BaseSolver> get_solver(const Problem &prob) {
  switch (prob.dim_c) {
  case 1:
    return make_shared<Solver<1>>();
  case 2:
    return make_shared<Solver<2>>();
  case 3:
    return make_shared<Solver<3>>();
  case 4:
    return make_shared<Solver<4>>();
  case 5:
    return make_shared<Solver<5>>();
  case 6:
    return make_shared<Solver<6>>();
  case 7:
    return make_shared<Solver<7>>();
  default:
    return make_shared<Solver<0>>();
  }
}

void run_t0() {
#ifdef PRINT_LOG
  auto start_t0 = chrono::high_resolution_clock::now();
#endif
  double x = 0.55;
  for (int i = 1; i <= 1000000; i++) {
    x = x + x;
    x = x / 2;
    x = x * x;
    x = sqrt(x);
    x = log(x);
    x = exp(x);
    x = x / (x + 2);
  }
#ifdef PRINT_LOG
  auto end_t0 = chrono::high_resolution_clock::now();
  cout << "\t"
       << "CPU Test ["
       << chrono::duration<double>(end_t0 - start_t0).count() * 1000 << "ms]"
       << endl;
#endif
}

void run_benchmark() {
  vector<vector<Group>> solutions = {{}};

  auto start_t0 = chrono::high_resolution_clock::now();
  run_t0();
  auto end_t0 = chrono::high_resolution_clock::now();
  double t0 = chrono::duration<double>(end_t0 - start_t0).count();
#ifdef PRINT_LOG
  cout << "T0 = " << t0 * 1000 << "ms" << endl;
#endif

  vector<double> runtimes = {t0};

  for (int prob_id = 1; prob_id <= 12; prob_id++) {
    Problem prob = get_default_problem(prob_id);
    shared_ptr<BaseSolver> solver = get_solver(prob);

#ifdef PRINT_LOG
    auto [num_areas, num_links] = prob.size();
    cout << "Problem " << prob_id << " (" << prob.dim_x << "×" << prob.dim_y
         << "): " << num_areas << " Nodes, " << num_links / 2 << " Edges"
         << endl;
#endif

    auto start_time = chrono::high_resolution_clock::now();
    vector<Group> groups = solver->solve(prob);
    auto end_time = chrono::high_resolution_clock::now();
    double runtime = chrono::duration<double>(end_time - start_time).count();

#ifdef PRINT_LOG
    cout << "T" << prob_id << " = " << runtime * 1000 << "ms, T" << prob_id
         << "/T0 = " << runtime / t0 << endl;
#endif

    solutions.push_back(move(groups));
    runtimes.push_back(runtime);
  }

  vector<string> roman = {"I",    "II", "III", "IV", "V",   "VI",   "VII",
                          "VIII", "IX", "X",   "XI", "XII", "XIII", "XIV"};

  /*
   * Table I
   */
  ofstream table_first("output/Table " + roman[0] + ".txt");
  table_first << "The test problem"
              << "\t"
              << "The pareto optimal paths"
              << "\t"
              << "The objective values"
              << "\t"
              << "Number of paths" << endl;

  for (int prob_id = 1; prob_id <= 12; prob_id++) {
    int prev_path_id = 0;
    for (const auto &[obj_cost, paths] : solutions[prob_id]) {
      table_first << "Test problem " << prob_id << "\t";
      table_first << "Path" << prob_id << "_" << prev_path_id + 1 << "_"
                  << prev_path_id + paths.size() << "\t";
      for (int i = 0; i < obj_cost.size(); i++) {
        table_first << obj_cost[i] << (i < obj_cost.size() - 1 ? "," : "\t");
      }
      table_first << paths.size() << endl;
      prev_path_id += (int)paths.size();
    }
  }

  table_first.close();

  /*
   * Tables II-XIII
   */

  for (int prob_id = 1; prob_id <= 12; prob_id++) {
    ofstream table("output/Table " + roman[prob_id] + ".txt");
    int max_len = 0;

    int path_id = 0;
    for (const auto &[obj_cost, paths] : solutions[prob_id]) {
      for (const Path &path : paths) {
        if (max_len < path.size()) {
          max_len = (int)path.size();
        }
        path_id++;
        table << (path_id == 1 ? "" : "\t");
        table << "Path" << prob_id << "_" << path_id << "(x)";
        table << "\t";
        table << "Path" << prob_id << "_" << path_id << "(y)";
      }
    }
    table << endl;

    for (int i = 0; i < max_len; i++) {
      path_id = 0;
      for (const auto &[obj_cost, paths] : solutions[prob_id]) {
        for (const Path &path : paths) {
          auto [x, y] = path[i];
          path_id++;
          table << (path_id == 1 ? "" : "\t");
          table << (i < path.size() ? x + 1 : 0);
          table << "\t";
          table << (i < path.size() ? y + 1 : 0);
        }
      }
      table << endl;
    }

    table.close();
  }

  /*
   * Table XIV
   */
  ofstream table_last("output/Table " + roman[13] + ".txt");
  table_last << "T0";
  for (int prob_id = 1; prob_id <= 12; prob_id++) {
    table_last << "\t"
               << "T" << prob_id << "/"
               << "T0";
  }
  table_last << endl;

  table_last << runtimes[0];
  cout << "T0 = " << runtimes[0] << endl;
  for (int prob_id = 1; prob_id <= 12; prob_id++) {
    table_last << "\t" << runtimes[prob_id] / runtimes[0];
    cout << "T" << prob_id << " = " << runtimes[prob_id] << endl;
  }
  table_last << endl;

  table_last.close();
}

void run_instance(const string &file, const vector<string> &obj_vector) {
  auto start_t0 = chrono::high_resolution_clock::now();
  run_t0();
  auto end_t0 = chrono::high_resolution_clock::now();
  double t0 = chrono::duration<double>(end_t0 - start_t0).count();
#ifdef PRINT_LOG
  cout << "T0 = " << t0 * 1000 << "ms" << endl;
#endif

  Problem prob = get_problem(file, obj_vector);
  auto solver = get_solver(prob);
#ifdef PRINT_LOG
  auto [num_areas, num_links] = prob.size();
  cout << "Problem (" << prob.dim_x << "x" << prob.dim_y << "): " << num_areas
       << " Nodes, " << num_links / 2 << " Edges" << endl;
#endif
  auto start_time = chrono::high_resolution_clock::now();
  vector<Group> groups = solver->solve(prob);
  auto end_time = chrono::high_resolution_clock::now();
  double runtime = chrono::duration<double>(end_time - start_time).count();
#ifdef PRINT_LOG
  cout << "T = " << runtime * 1000 << "ms (" << runtime / t0 << ")" << endl;
#endif

  /*
   * Result I
   */
  ofstream table_first("output/Result I.txt");
  table_first << "The test Problem"
              << "\t"
              << "The pareto optimal paths"
              << "\t"
              << "The Objective values"
              << "\t"
              << "Number of paths" << endl;

  int prev_path_id = 0;
  for (const auto &[obj_cost, paths] : groups) {
    table_first << "Test Problem"
                << "\t";
    table_first << "Path_" << prev_path_id + 1 << "_"
                << prev_path_id + paths.size() << "\t";
    for (int i = 0; i < obj_cost.size(); i++) {
      table_first << obj_cost[i] << (i < obj_cost.size() - 1 ? "," : "\t");
    }
    table_first << paths.size() << endl;
    prev_path_id += (int)paths.size();
  }

  table_first.close();

  /*
   * Result II
   */

  ofstream table("output/Result II.txt");
  int max_len = 0;

  int path_id = 0;
  for (const auto &[obj_cost, paths] : groups) {
    for (const Path &path : paths) {
      if (max_len < path.size()) {
        max_len = (int)path.size();
      }
      path_id++;
      table << (path_id == 1 ? "" : "\t");
      table << "Path_" << path_id << "(x)";
      table << "\t";
      table << "Path_" << path_id << "(y)";
    }
  }
  table << endl;

  for (int i = 0; i < max_len; i++) {
    path_id = 0;
    for (const auto &[obj_cost, paths] : groups) {
      for (const Path &path : paths) {
        auto [x, y] = path[i];
        path_id++;
        table << (path_id == 1 ? "" : "\t");
        table << (i < path.size() ? x + 1 : 0);
        table << "\t";
        table << (i < path.size() ? y + 1 : 0);
      }
    }
    table << endl;
  }

  table.close();

  /*
   * Result III
   */
  ofstream table_last("output/Result III.txt");
  table_last << "T0"
             << "\t"
             << "T/T0" << endl;

  table_last << t0 << "\t" << runtime / t0 << endl;

  table_last.close();
}

int main(int argc, char **argv) {
  if (argc == 1) {
    run_benchmark();
  } else {
    run_instance(argv[1], vector<string>(argv + 2, argv + argc));
  }
  return 0;
}
