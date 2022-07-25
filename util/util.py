import casadi as cs

def quad_form(A: cs.SX | cs.DM, x: cs.SX | cs.DM) -> cs.SX | cs.DM:
    '''Calculates quadratic form x^T A x.'''
    if A.is_vector():
        A = cs.diag(A)
    return cs.bilin(A, x, x)
