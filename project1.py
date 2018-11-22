import sys
import networkx as nx
import pandas as pd
import numpy as np
import collections
import math
import graphviz
from graphviz import Source
from networkx.drawing.nx_pydot import write_dot




def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in nx.edges(dag):
            print('edge: ' + str(edge))
            f.write("{}, {}\n".format(edge[0], edge[1]))

def find_k(data, cols): #list: k's for each x_i (index i)
    res = collections.defaultdict(int)
    for i in range(cols):
        column = data.iloc[:, i]
        res[i] = column.nunique()
    return res


def get_m(data, graph):
    node_list = list(data.columns.values)
    nodes = {node_list[i]:i for i in range(0, len(node_list))} #name -> column index
    m_ijk =  m_ij0 = collections.defaultdict(int)
    n_rows, n_cols = data.shape[0], data.shape[1]
    # print('rows and columns: ' + str(n_rows) + ' , ' + str(n_cols))
    a_ij0 = find_k(data, n_cols) #dict: x_i -> how many values it takes on

    for m in range(n_rows): #iterate over samples
        sample = data.iloc[m]
        for n in range(n_cols): #iterate over features
            cur_val = data.iloc[m, n]
            # print('cur val : ' + str(cur_val))
            node = node_list[n] #feature name
            # print('cur node: ' + str(node))
            parents = graph.predecessors(node)
            # print('parents of node' + str(node) +': ' + str(parents))
            multi_index = []
            for parent in parents:
                parent_ind = nodes[parent]
                parent_val = data.iloc[ m,parent_ind ] #parent's instantiation
                multi_index.append(parent_val)
            m_ijk[ (n, tuple(multi_index) , cur_val) ] += 1
            m_ij0[ (n, tuple(multi_index)) ] += 1
    # print('m_ijk: ' + str(m_ijk))
    return m_ijk, m_ij0, a_ij0


def bayes(m, m_sum, priors):
    score = 0
    for key in priors: #summing over n
        alpha = priors[key]
        for inst in m_sum: #summing over parental instantiations
            score += math.lgamma(alpha) - math.lgamma(alpha + m_sum[inst])
    for key in m: #summing over all ijk's
        score +=  math.lgamma(1+ m[key]) - math.lgamma(1)
    return score


def k2_search(data, ind2name, outfile):
    node_list = list(data.columns.values)
    nodes = {i : node_list[i] for i in range(0, len(node_list))}
    print('Initial state: making a sparse graph')
    G = nx.DiGraph() #graph w/ no edges
    G.add_nodes_from(node_list)
    print('here are g\'s nodes: ' + str( nx.nodes(G)))
    print('G\'s edges: '+ str(nx.edges(G)))
    m, m_sum, priors = get_m(data, G)
    orig_score = best_score = bayes(m, m_sum, priors) #score for a sparse graph is
    print('score of sparse graph: ')
    print(str(best_score))
    print('start k2 algo')
    for i in range(len(nodes)):
        for j in range(len(nodes)): #add parents to node i until max Bayes score
            if i==j or G.has_edge(nodes[j],nodes[i]) or G.has_edge(nodes[i],nodes[j]):
                continue
            H = G
            H.add_edge(nodes[j], nodes[i])
            # print('current graph H has edges: ' + str(nx.edges(H)))
            if nx.is_directed_acyclic_graph(H):
                m_counts, m_sums, priors = get_m(data, H)
                cur_score = bayes(m_counts, m_sum, priors)
                # print('this H has score: ' + str(cur_score))
                if cur_score > best_score:
                    # print('There is a new best score: ' + str(cur_score))
                    best_score = cur_score
                    G = H
            else:
                H.remove_edge(nodes[j], nodes[i])

            if best_score <= orig_score:
                break
    # print('found a best G:')
    return G


def compute(infile, outfile):
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
    data = pd.read_csv(infile, encoding = 'utf-8')
    print('data after loading:')
    print(data.head())

    nodes = list(data.columns.values)
    ind2name = {i : nodes[i] for i in range(0, len(nodes))}
    graph = k2_search(data, ind2name, outfile)

    print('writing G to gph...')
    write_gph(graph, ind2name, outfile)

    print('drawing it now')
    pos = nx.nx_agraph.graphviz_layout(graph)
    nx.draw(graph, pos=pos)
    path = infile+'.dot'
    write_dot(graph, path)
    s = Source.from_file(path)
    s.view()

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
