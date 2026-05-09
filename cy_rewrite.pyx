# cython: language_level=3
"""Cython-compiled unified_rewrite — drop-in replacement for RewriteContext.unified_rewrite."""
import collections
from tinygrad.uop.ops import UOp, Ops, unwrap, BottomUpGate

cdef object SENTINEL = object()
cdef set CALL_OPS = {Ops.CALL, Ops.FUNCTION}

def cy_rewrite(self, uop, ctx=None):
    """Cython-compiled PatternMatcher.rewrite — bitmask early-reject + move-toward-front on hit."""
    cdef list pats = self._plist[uop.op]
    if pats is None:
        return None
    if self._use_mega:
        if uop.op in self._mega:
            mega = self._mega[uop.op]
            if mega is not None:
                ret = mega(uop, ctx)
                if ret is not None and ret is not uop: return ret
                return None
        elif len(pats) >= 2:
            from tinygrad.uop.upat import mega_compile
            mega = mega_compile(tuple((e[0], self._fxn_map[id(e[0])]) for e in pats))
            self._mega[uop.op] = mega
            if mega is not None:
                ret = mega(uop, ctx)
                if ret is not None and ret is not uop: return ret
                return None
    cdef dict uop_dict = uop.__dict__
    cached = uop_dict.get('_src_ops_mask')
    cdef object ler
    if cached is None:
        ler = 0
        for u in uop.src:
            ler |= 1 << int(u.op)
        uop_dict['_src_ops_mask'] = ler
    else:
        ler = cached
    cdef int i
    cdef int n = len(pats)
    cdef list entry
    cdef object er
    for i in range(n):
        entry = <list>pats[i]
        er = entry[2]
        if (er & ler) != er:
            continue
        ret = (<object>entry[1])(uop, ctx)
        if ret is not None and ret is not uop:
            return ret
    return None

def cy_unified_rewrite(self, root):
    cdef dict replace = self.replace
    stack = collections.deque([(root, 0, root)])
    cdef set on_stack = {root}
    cdef dict waitlist = {}
    cdef int stage
    cdef tuple new_src
    cdef list tmp
    cdef bint enter_calls = self.enter_calls
    pm = self.pm
    bpm = self.bpm
    ctx = self.ctx
    cdef dict bpm_cache = self._bpm_cache if bpm is not None else {}

    while stack:
        if len(stack) > 250000:
            raise RuntimeError("infinite loop in graph_rewrite (stack too big)")
        n, stage, new_n = stack.pop()
        if n in replace:
            continue
        if stage == 0:
            if bpm is not None:
                test_n = n
                seen = set()
                gate = False
                while test_n is not None:
                    if test_n in seen:
                        raise RuntimeError("infinite loop in fixed_point_rewrite")
                    seen.add(test_n)
                    new_n = test_n
                    try:
                        test_n = self.cached_bpm_rewrite(test_n)
                    except BottomUpGate:
                        replace[n] = unwrap(test_n)
                        if n in waitlist:
                            stack.extend(waitlist.pop(n))
                        gate = True
                        break
                if gate:
                    continue
            stack.append((n, 1, new_n))
            if not enter_calls and new_n.op in CALL_OPS:
                replace[new_n.src[0]] = new_n.src[0]
            for x in reversed(new_n.src):
                if x in on_stack:
                    continue
                stack.append((x, 0, x))
                on_stack.add(x)
        elif stage == 1:
            tmp = []
            broke = False
            for x in new_n.src:
                rx = replace.get(x, SENTINEL)
                if rx is SENTINEL:
                    waitlist.setdefault(x, []).append((n, 1, new_n))
                    broke = True
                    break
                tmp.append(rx)
            if broke:
                continue
            new_src = tuple(tmp)
            if new_src == new_n.src:
                if pm is None:
                    replace[n] = new_n
                    if n in waitlist:
                        stack.extend(waitlist.pop(n))
                    continue
                new_src_n = pm.rewrite(new_n, ctx)
                if new_src_n is None:
                    replace[n] = new_n
                    if n in waitlist:
                        stack.extend(waitlist.pop(n))
                    continue
            else:
                new_src_n = UOp(new_n.op, new_n.dtype, new_src, new_n.arg, new_n.tag)
            stack.append((n, 2, new_src_n))
            stack.append((new_src_n, 0, new_src_n))
        else:
            replaced_new_n = replace.get(new_n, SENTINEL)
            if replaced_new_n is SENTINEL:
                waitlist.setdefault(new_n, []).append((n, 2, new_n))
            else:
                replace[n] = replaced_new_n
                if n in waitlist:
                    stack.extend(waitlist.pop(n))
    return replace[root]
