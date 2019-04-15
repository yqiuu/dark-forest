#include"halotree.h"


int NodeVector_Append(NodeVector *vector, double delta, double mass) {
    if (vector->n_node == vector->n_max) {
        vector->n_max *= 2;
        vector->data = realloc(vector->data, vector->n_max*sizeof(Node));
    }
    int idx = (vector->n_node)++;
    Node* node = vector->data + idx;
    node->trace = delta;
    node->mass = mass;
    node->idx_tail = -1;
    node->idx_next = -1;
    return idx;
}


NodeVector NodeVector_Init(int n_base) {
    NodeVector vector;
    vector.n_max = n_base;
    vector.n_node = 0;
    vector.data = (Node*)malloc(n_base*sizeof(Node));
    return vector;
}


void NodeVector_Free(NodeVector *vector) {
    free(vector->data);
}
