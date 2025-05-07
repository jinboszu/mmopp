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

#include <fstream>
#include <set>
using namespace std;

#include "nlohmann/json.hpp"
using namespace nlohmann;

#include "pareto/front.h"
using namespace pareto;

// #define PRINT_LOG

#define within(x, y, X, Y) ((x) >= (0) && (x) < (X) && (y) >= (0) && (y) < (Y))

template <typename T> using Matrix = vector<vector<T>>;

using Coord = pair<int, int>;

vector<Coord> steps = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};

struct Problem {
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
  vector<double> scale_factors;

  Coord size() const { return size(passable); }

  Coord size(const Matrix<bool> &retained) const {
    int num_areas = 0;
    int num_links = 0;
    for (int x = 0; x < dim_x; x++) {
      for (int y = 0; y < dim_y; y++) {
        if (retained[x][y]) {
          num_areas++;
          for (auto [step_x, step_y] : steps) {
            int next_x = x + step_x;
            int next_y = y + step_y;
            if (within(next_x, next_y, dim_x, dim_y) &&
                retained[next_x][next_y]) {
              num_links++;
            }
          }
        }
      }
    }
    return {num_areas, num_links};
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
  int dim_x = static_cast<int>(MAP[0].size());
  int dim_y = static_cast<int>(MAP.size());

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
    auto YELLOW_AREAS = data["Yellow_areas"].get<set<Coord>>();
    YELLOW_AREAS.erase({start_x + 1, start_y + 1});
    YELLOW_AREAS.erase({goal_x + 1, goal_y + 1});
    for (auto [X, Y] : YELLOW_AREAS) {
      key_list.emplace_back(X - 1, Y - 1);
      key_matrix[X - 1][Y - 1] = dim_k++;
    }
  }

  Matrix<vector<double>> cost_matrix(dim_x, vector<vector<double>>(dim_y));
  int dim_c = 0;
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
      auto RED_AREAS = data["Red_areas"].get<set<Coord>>();
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
            int degree = 0;
            for (auto [step_x, step_y] : steps) {
              int next_x = x + step_x;
              int next_y = y + step_y;
              if (within(next_x, next_y, dim_x, dim_y) &&
                  passable[next_x][next_y]) {
                degree++;
              }
            }
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
      int dim_f = static_cast<int>(F[0].size()) - 2;
      map<Coord, vector<double>> F_MAP;
      for (const vector<double> &row : F) {
        F_MAP[{lround(row[0]), lround(row[1])}] =
            vector<double>(row.begin() + 2, row.end());
      }
      for (int x = 0; x < dim_x; x++) {
        for (int y = 0; y < dim_y; y++) {
          if (passable[x][y]) {
            if (auto it = F_MAP.find({x + 1, y + 1}); it != F_MAP.end()) {
              for (double c : it->second) {
                cost_matrix[x][y].push_back(c);
              }
            } else {
              for (int f = 0; f < dim_f; f++) {
                cost_matrix[x][y].push_back(0);
              }
            }
          }
        }
      }
      dim_c += dim_f;
    }
  }

  vector<double> scale_factors(dim_c, 1);
  for (int x = 0; x < dim_x; x++) {
    for (int y = 0; y < dim_y; y++) {
      if (passable[x][y]) {
        for (int c = 0; c < dim_c; c++) {
          while (abs(cost_matrix[x][y][c] * scale_factors[c] -
                     round(cost_matrix[x][y][c] * scale_factors[c])) >=
                 1e-6 * scale_factors[c]) {
            scale_factors[c] *= 10;
          }
        }
      }
    }
  }
  for (int x = 0; x < dim_x; x++) {
    for (int y = 0; y < dim_y; y++) {
      if (passable[x][y]) {
        for (int c = 0; c < dim_c; c++) {
          cost_matrix[x][y][c] = round(cost_matrix[x][y][c] * scale_factors[c]);
        }
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
          move(scale_factors)};
}

Problem get_default_problem(int prob_id) {
  json data;
  ifstream ifs("data/Problem_" + to_string(prob_id) + ".json");
  ifs >> data;
  ifs.close();

  return parse_problem(data, get_default_objs(prob_id));
}

Problem get_problem(const string &file, const vector<string> &args) {
  json data;
  ifstream ifs(file);
  ifs >> data;
  ifs.close();

  vector<Objective> objs;
  for (const string &arg : args) {
    if (Objective obj = parse_objective(arg); obj != unknown) {
      objs.push_back(obj);
    }
  }

  return parse_problem(data, objs);
}

using Path = vector<Coord>;
using Solution = vector<pair<vector<double>, vector<Path>>>;

struct Solver {
  virtual ~Solver() = default;
  virtual Solution solve(const Problem &prob) const = 0;
};

template <size_t M = 0> struct SolverImpl : Solver {
  using Cost = point<double, M>;

  struct Edge {
    int prev_node;
    int next_node;
    Cost edge_cost;
    Path edge_pass;
  };

  Solution solve(const Problem &prob) const override {
#ifdef PRINT_LOG
    auto time_point_0 = chrono::high_resolution_clock::now();
#endif

    const auto &[start_x, start_y, goal_x, goal_y, dim_x, dim_y, dim_k, dim_c,
                 passable, key_list, key_matrix, cost_matrix, scale_factors] =
        prob;

    /*
     * Map Reduction
     */
    Matrix<int> depth(dim_x, vector<int>(dim_y, -1));
    Matrix<int> low(dim_x, vector<int>(dim_y, -1));
    Matrix<bool> flag(dim_x, vector<bool>(dim_y, false));
    Matrix<vector<Coord>> children(dim_x, vector<vector<Coord>>(dim_y));
    Matrix<bool> retained(dim_x, vector<bool>(dim_y, false));

    function<void(int, int, int)> build_tree;
    build_tree = [&](int x, int y, int d) {
      depth[x][y] = d;
      low[x][y] = d;
      flag[x][y] = x == goal_x && y == goal_y || key_matrix[x][y] != -1;

      for (auto [step_x, step_y] : steps) {
        int next_x = x + step_x;
        int next_y = y + step_y;
        if (within(next_x, next_y, dim_x, dim_y) && passable[next_x][next_y]) {
          if (depth[next_x][next_y] == -1) {
            children[x][y].emplace_back(next_x, next_y);
            build_tree(next_x, next_y, d + 1);
            if (low[x][y] > low[next_x][next_y]) {
              low[x][y] = low[next_x][next_y];
            }
            if (flag[next_x][next_y]) {
              flag[x][y] = true;
            }
          } else if (low[x][y] > depth[next_x][next_y]) {
            low[x][y] = depth[next_x][next_y];
          }
        }
      }
    };

    function<void(int, int)> trim_tree;
    trim_tree = [&](int x, int y) {
      retained[x][y] = true;
      for (auto [child_x, child_y] : children[x][y]) {
        if (low[child_x][child_y] < depth[x][y] || flag[child_x][child_y]) {
          trim_tree(child_x, child_y);
        }
      }
    };

    bool solvable = false;
    if (passable[start_x][start_y]) {
      build_tree(start_x, start_y, 0);
      trim_tree(start_x, start_y);
      if (retained[goal_x][goal_y]) {
        solvable = true;
        for (int k = 0; k < dim_k; k++) {
          auto [key_x, key_y] = key_list[k];
          if (!retained[key_x][key_y]) {
            solvable = false;
            break;
          }
        }
      }
    }

    if (!solvable) {
#ifdef PRINT_LOG
      cout << "\t" << "Unsolvable!" << endl;
#endif
      return {};
    }

    Matrix<int> degree_matrix(dim_x, vector<int>(dim_y, 0));
    for (int x = 0; x < dim_x; x++) {
      for (int y = 0; y < dim_y; y++) {
        if (retained[x][y]) {
          for (auto [step_x, step_y] : steps) {
            int next_x = x + step_x;
            int next_y = y + step_y;
            if (within(next_x, next_y, dim_x, dim_y) &&
                retained[next_x][next_y]) {
              degree_matrix[next_x][next_y]++;
            }
          }
        }
      }
    }

#ifdef PRINT_LOG
    auto time_point_1 = chrono::high_resolution_clock::now();
    auto [num_areas, num_links] = prob.size(retained);
    cout << "\t" << "Map Reduction ("
         << chrono::duration<double>(time_point_1 - time_point_0).count() * 1000
         << " ms): " << num_areas << " Nodes, " << num_links / 2 << " Edges"
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
      for (auto [step_x, step_y] : steps) {
        int curr_x = x + step_x;
        int curr_y = y + step_y;
        if (within(curr_x, curr_y, dim_x, dim_y) && retained[curr_x][curr_y]) {
          Cost edge_cost(dim_c, 0);
          Path edge_pass;
          int prev_x = x;
          int prev_y = y;
          while (node_matrix[curr_x][curr_y] == -1) {
            for (int c = 0; c < dim_c; c++) {
              edge_cost[c] += cost_matrix[curr_x][curr_y][c];
            }
            edge_pass.emplace_back(curr_x, curr_y);

            for (auto [next_step_x, next_step_y] : steps) {
              int next_x = curr_x + next_step_x;
              int next_y = curr_y + next_step_y;
              if ((next_x != prev_x || next_y != prev_y) &&
                  within(next_x, next_y, dim_x, dim_y) &&
                  retained[next_x][next_y]) {
                prev_x = curr_x;
                prev_y = curr_y;
                curr_x = next_x;
                curr_y = next_y;
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
    cout << "\t" << "Graph Model ("
         << chrono::duration<double>(time_point_2 - time_point_1).count() * 1000
         << " ms): " << dim_n << " Nodes, " << edges.size() / 2 << " Edges"
         << endl;
#endif

    /*
     * Ideal Costs
     */
    Matrix<Cost> ideal_to_keys(dim_n, vector<Cost>(dim_k, Cost(dim_c, 1e9)));
    for (int c = 0; c < dim_c; c++) {
      for (int k = 0; k < dim_k; k++) {
        ideal_to_keys[key_nodes[k]][k][c] = 0;
        set<pair<double, int>> pq = {{0, key_nodes[k]}};
        while (!pq.empty()) {
          auto [cost, node] = *pq.begin();
          pq.erase(pq.begin());
          for (int edge_id : in_edges[node]) {
            int prev_node = edges[edge_id].prev_node;
            double prev_cost = edges[edge_id].edge_cost[c] + cost;
            if (node != key_nodes[k]) {
              prev_cost += costs[node][c];
            }
            if (prev_cost < ideal_to_keys[prev_node][k][c]) {
              if (ideal_to_keys[prev_node][k][c] != 1e9) {
                pq.erase({ideal_to_keys[prev_node][k][c], prev_node});
              }
              ideal_to_keys[prev_node][k][c] = prev_cost;
              pq.emplace(prev_cost, prev_node);
            }
          }
        }
      }
    }

    vector<Cost> ideal_to_goal(dim_n, Cost(dim_c, 1e9));
    for (int c = 0; c < dim_c; c++) {
      ideal_to_goal[goal_node][c] = 0;
      set<pair<double, int>> pq = {{0, goal_node}};
      while (!pq.empty()) {
        auto [cost, node] = *pq.begin();
        pq.erase(pq.begin());
        for (int edge_id : in_edges[node]) {
          int prev_node = edges[edge_id].prev_node;
          double prev_cost = edges[edge_id].edge_cost[c] + cost;
          if (node != goal_node) {
            prev_cost += costs[node][c];
          }
          if (prev_cost < ideal_to_goal[prev_node][c]) {
            if (ideal_to_goal[prev_node][c] != 1e9) {
              pq.erase({ideal_to_goal[prev_node][c], prev_node});
            }
            ideal_to_goal[prev_node][c] = prev_cost;
            pq.emplace(prev_cost, prev_node);
          }
        }
      }
    }

    /*
     * Herustic Function
     */
    auto heuristic = [&, heu_mem = vector<map<int, Cost>>(dim_n),
                      mst_mem = map<int, Cost>()](int node,
                                                  int status) mutable {
      if (auto it = heu_mem[node].find(status); it != heu_mem[node].end()) {
        return it->second;
      }

      if (status == (1 << dim_k) - 1) {
        if (node == goal_node) {
          return heu_mem[node][status] = ideal_to_goal[node];
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

          lb += mst_mem[status] = move(mst_cost);
        }
      }

      return heu_mem[node][status] = move(lb);
    };

    /*
     * Best-First Search
     */
    struct Previous {
      int prev_node;
      int prev_status;
      Cost prev_cost;
      int edge_id;
    };

    struct Extra {
      Cost curr_est;
      vector<Previous> previous;
    };

    using Front = front<double, M, Extra>;

    struct Quadruple {
      int node;
      int status;
      Cost cost;
      Cost est;
    };

    struct QuadrupleComparator {
      bool operator()(const Quadruple &a, const Quadruple &b) const {
        if (a.est != b.est) {
          return lexicographical_compare(a.est.begin(), a.est.end(),
                                         b.est.begin(), b.est.end());
        } else if (a.node != b.node) {
          return a.node < b.node;
        } else if (a.status != b.status) {
          return a.status < b.status;
        } else {
          return lexicographical_compare(a.cost.begin(), a.cost.end(),
                                         b.cost.begin(), b.cost.end());
        }
      }
    };

    using Queue = set<Quadruple, QuadrupleComparator>;

    vector<map<int, Front>> tentative(dim_n);
    const Front &goal = tentative[goal_node][(1 << dim_k) - 1];

    Cost start_cost = costs[start_node];
    Cost start_est = start_cost + heuristic(start_node, 0);
    tentative[start_node][0][start_cost] = {start_est, {}};

    Queue open = {{start_node, 0, move(start_cost), move(start_est)}};

    int iteration = 0;
    while (!open.empty()) {
      iteration++;
      auto [node, status, cost, est] = *open.begin();
      open.erase(open.begin());

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

        if (auto it = next_tent.find(next_cost); it != next_tent.end()) {
          it->second.previous.push_back({node, status, cost, edge_id});
          continue;
        }

        if (next_tent.dominates(next_cost)) {
          continue;
        }

        if (auto it_dom = next_tent.find_dominated(next_cost);
            it_dom != next_tent.end()) {
          vector<Cost> dominated;
          while (it_dom != next_tent.end()) {
            const Cost &dom_cost = it_dom->first;
            const Cost &dom_est = it_dom->second.curr_est;
            if (open.find({next_node, next_status, dom_cost, dom_est}) ==
                open.end()) {
              cout << "very bad 1" << endl;
            }
            open.erase({next_node, next_status, dom_cost, dom_est});
            dominated.push_back(dom_cost);
            ++it_dom;
          }
          for (const Cost &dom_cost : dominated) {
            next_tent.erase(dom_cost);
          }
        }

        Cost next_est = next_cost + heuristic(next_node, next_status);
        if (goal.dominates(next_est)) {
          continue;
        }

        next_tent[next_cost] = {next_est, {{node, status, cost, edge_id}}};
        open.insert({next_node, next_status, move(next_cost), move(next_est)});
      }
    }

#ifdef PRINT_LOG
    auto time_point_3 = chrono::high_resolution_clock::now();
    cout << "\t" << "Best-First Search ("
         << chrono::duration<double>(time_point_3 - time_point_2).count() * 1000
         << " ms): " << iteration << " Iterations" << endl;
#endif

    /*
     * Path Construction
     */
    int num_paths = 0;
    Solution solution;
    for (const auto &[goal_cost, goal_extra] : goal) {
      vector<Path> paths;

      function<void(int, int, const Cost &)> backtrack;
      backtrack = [&, rev_path = Path(), visited = vector<set<int>>(dim_n)](
                      int node, int status, const Cost &cost) mutable {
        rev_path.push_back(coords[node]);
        visited[node].insert(status);

        if (node == start_node && status == 0) {
          paths.emplace_back(rev_path.rbegin(), rev_path.rend());
        } else {
          int rev_path_size = static_cast<int>(rev_path.size());
          for (const auto &[prev_node, prev_status, prev_cost, edge_id] :
               tentative[node][status][cost].previous) {
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

      sort(paths.begin(), paths.end());

      vector<double> cost(goal_cost.begin(), goal_cost.end());
      for (int c = 0; c < dim_c; c++) {
        cost[c] /= scale_factors[c];
      }

      num_paths += static_cast<int>(paths.size());
      solution.emplace_back(move(cost), move(paths));
    }

    sort(solution.begin(), solution.end());

#ifdef PRINT_LOG
    auto time_point_4 = chrono::high_resolution_clock::now();
    cout << "\t" << "Path Construction ("
         << chrono::duration<double>(time_point_4 - time_point_3).count() * 1000
         << " ms): " << solution.size() << " Points, " << num_paths << " Paths"
         << endl;
    cout << "\t" << "Total Time: "
         << chrono::duration<double>(time_point_4 - time_point_0).count() * 1000
         << " ms" << endl;
#endif

    return solution;
  }
};

unique_ptr<Solver> get_solver(const Problem &prob) {
  switch (prob.dim_c) {
  case 1:
    return make_unique<SolverImpl<1>>();
  case 2:
    return make_unique<SolverImpl<2>>();
  case 3:
    return make_unique<SolverImpl<3>>();
  case 4:
    return make_unique<SolverImpl<4>>();
  case 5:
    return make_unique<SolverImpl<5>>();
  case 6:
    return make_unique<SolverImpl<6>>();
  case 7:
    return make_unique<SolverImpl<7>>();
  default:
    return make_unique<SolverImpl<0>>();
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
  cout << "CPU Test" << endl;
  cout << "\t" << "Total Time: "
       << chrono::duration<double>(end_t0 - start_t0).count() * 1000 << " ms"
       << endl;
#endif
}

void run_benchmark() {
  vector<Solution> solutions = {{}};

  auto start_t0 = chrono::high_resolution_clock::now();
  run_t0();
  auto end_t0 = chrono::high_resolution_clock::now();
  double t0 = chrono::duration<double>(end_t0 - start_t0).count();
#ifdef PRINT_LOG
  cout << "T0 = " << t0 * 1000 << " ms" << endl;
#endif

  vector<double> runtimes = {t0};

  for (int prob_id = 1; prob_id <= 12; prob_id++) {
    Problem prob = get_default_problem(prob_id);
    auto solver = get_solver(prob);

#ifdef PRINT_LOG
    auto [num_areas, num_links] = prob.size();
    cout << "Problem " << prob_id << " (" << prob.dim_x << "×" << prob.dim_y
         << "): " << num_areas << " Nodes, " << num_links / 2 << " Edges"
         << endl;
#endif

    auto start_time = chrono::high_resolution_clock::now();
    Solution solution = solver->solve(prob);
    auto end_time = chrono::high_resolution_clock::now();
    double runtime = chrono::duration<double>(end_time - start_time).count();

#ifdef PRINT_LOG
    cout << "T" << prob_id << " = " << runtime * 1000 << " ms, T" << prob_id
         << "/T0 = " << runtime / t0 << endl;
#endif

    solutions.push_back(move(solution));
    runtimes.push_back(runtime);
  }

  vector<string> roman = {"I",    "II", "III", "IV", "V",   "VI",   "VII",
                          "VIII", "IX", "X",   "XI", "XII", "XIII", "XIV"};

  /*
   * Table I
   */
  ofstream table_first("results/Table " + roman[0] + ".txt");
  table_first << "The test problem" << "\t";
  table_first << "The pareto optimal paths" << "\t";
  table_first << "The objective values" << "\t";
  table_first << "Number of paths" << endl;

  for (int prob_id = 1; prob_id <= 12; prob_id++) {
    int path_count = 0;
    for (const auto &[cost, paths] : solutions[prob_id]) {
      table_first << "Test problem " << prob_id << "\t";
      table_first << "Path" << prob_id << "_" << path_count + 1 << "_"
                  << path_count + paths.size() << "\t";
      for (int i = 0; i < cost.size(); i++) {
        table_first << cost[i] << (i < cost.size() - 1 ? "," : "\t");
      }
      table_first << paths.size() << endl;
      path_count += static_cast<int>(paths.size());
    }
  }

  table_first.close();

  /*
   * Tables II-XIII
   */

  for (int prob_id = 1; prob_id <= 12; prob_id++) {
    ofstream table("results/Table " + roman[prob_id] + ".txt");
    int max_len = 0;

    int path_id = 0;
    for (const auto &[cost, paths] : solutions[prob_id]) {
      for (const Path &path : paths) {
        if (max_len < path.size()) {
          max_len = static_cast<int>(path.size());
        }
        path_id++;
        table << (path_id == 1 ? "" : "\t");
        table << "Path" << prob_id << "_" << path_id << "(x)" << "\t";
        table << "Path" << prob_id << "_" << path_id << "(y)";
      }
    }
    table << endl;

    for (int i = 0; i < max_len; i++) {
      path_id = 0;
      for (const auto &[cost, paths] : solutions[prob_id]) {
        for (const Path &path : paths) {
          auto [x, y] = path[i];
          path_id++;
          table << (path_id == 1 ? "" : "\t");
          table << (i < path.size() ? x + 1 : 0) << "\t";
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
  ofstream table_last("results/Table " + roman[13] + ".txt");
  table_last << "T0";
  for (int prob_id = 1; prob_id <= 12; prob_id++) {
    table_last << "\t" << "T" << prob_id << "/" << "T0";
  }
  table_last << endl;

  table_last << runtimes[0];
  for (int prob_id = 1; prob_id <= 12; prob_id++) {
    table_last << "\t" << runtimes[prob_id] / runtimes[0];
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
  cout << "T0 = " << t0 * 1000 << " ms" << endl;
#endif

  Problem prob = get_problem(file, obj_vector);
  auto solver = get_solver(prob);
#ifdef PRINT_LOG
  auto [num_areas, num_links] = prob.size();
  cout << "Problem (" << prob.dim_x << "x" << prob.dim_y << "): " << num_areas
       << " Nodes, " << num_links / 2 << " Edges" << endl;
#endif
  auto start_time = chrono::high_resolution_clock::now();
  Solution solution = solver->solve(prob);
  auto end_time = chrono::high_resolution_clock::now();
  double runtime = chrono::duration<double>(end_time - start_time).count();
#ifdef PRINT_LOG
  cout << "T = " << runtime * 1000 << " ms (" << runtime / t0 << ")" << endl;
#endif

  /*
   * Result I
   */
  ofstream table_first("results/Result I.txt");
  table_first << "The test Problem" << "\t";
  table_first << "The pareto optimal paths" << "\t";
  table_first << "The Objective values" << "\t";
  table_first << "Number of paths" << endl;

  int path_count = 0;
  for (const auto &[cost, paths] : solution) {
    table_first << "Test Problem" << "\t";
    table_first << "Path_" << path_count + 1 << "_" << path_count + paths.size()
                << "\t";
    for (int i = 0; i < cost.size(); i++) {
      table_first << cost[i] << (i < cost.size() - 1 ? "," : "\t");
    }
    table_first << paths.size() << endl;
    path_count += static_cast<int>(paths.size());
  }

  table_first.close();

  /*
   * Result II
   */

  ofstream table("results/Result II.txt");
  int max_len = 0;

  int path_id = 0;
  for (const auto &[cost, paths] : solution) {
    for (const Path &path : paths) {
      if (max_len < path.size()) {
        max_len = static_cast<int>(path.size());
      }
      path_id++;
      table << (path_id == 1 ? "" : "\t");
      table << "Path_" << path_id << "(x)" << "\t";
      table << "Path_" << path_id << "(y)";
    }
  }
  table << endl;

  for (int i = 0; i < max_len; i++) {
    path_id = 0;
    for (const auto &[cost, paths] : solution) {
      for (const Path &path : paths) {
        auto [x, y] = path[i];
        path_id++;
        table << (path_id == 1 ? "" : "\t");
        table << (i < path.size() ? x + 1 : 0) << "\t";
        table << (i < path.size() ? y + 1 : 0);
      }
    }
    table << endl;
  }

  table.close();

  /*
   * Result III
   */
  ofstream table_last("results/Result III.txt");
  table_last << "T0" << "\t";
  table_last << "T/T0" << endl;

  table_last << t0 << "\t";
  table_last << runtime / t0 << endl;

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
