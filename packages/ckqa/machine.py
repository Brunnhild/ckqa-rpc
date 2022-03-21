# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle

FAIL_STATE = -1
ROOT_STATE = 1

RESIZE_DELTA = 64
END_NODE_BASE = -1
ROOT_NODE_BASE = 1
ROOT_NODE_INDEX = 0


class Trie:

    def __init__(self):
        self.base = np.zeros(1, int)
        self.check = np.zeros(1, int)

    def __getitem__(self, keyword):
        b = self.base[0]
        for character in keyword:
            p = b + ord(character) + 1

            if p >= len(self.check):
                return False

            if b == self.check[p]:
                b = self.base[p]
            else:
                return False

        p = b
        n = self.base[p]
        if b == self.check[p] and n < 0:
            return True

        return False

    def serialize(self, path):

        if not os.path.isdir(path):
            os.mkdir(path)

        np.save(os.path.join(path, 'base.npy'), self.base)
        np.save(os.path.join(path, 'check.npy'), self.check)

    def deserialize(self, path):

        self.base = np.load(os.path.join(path, 'base.npy'))
        self.check = np.load(os.path.join(path, 'check.npy'))


class State:
    __slots__ = ('code', 'depth', 'left', 'right', 'index', 'base', 'sub_key', 'children')

    def __init__(self):
        self.code = 0
        self.depth = 0
        self.left = 0
        self.right = 0
        self.index = 0
        self.base = 0

        self.sub_key = []
        self.children = []


class TrieBuilder:

    def __init__(self):
        self.trie = Trie()
        self.output = {}

        self.used = np.zeros(0, bool)
        self.next_check_pos = None
        self.key = []

        self.llt = State()

    def add_keywords_from_list(self, keywords):
        self._resize(RESIZE_DELTA)

        for keyword in keywords:
            self.key.append(keyword)

        self.key = sorted(self.key)

        self.trie.base[0] = ROOT_NODE_BASE
        self.next_check_pos = 0

        self.llt.depth = 0
        self.llt.left = 0
        self.llt.right = len(self.key)
        self.llt.sub_key = []
        self.llt.children = []
        self.llt.index = ROOT_NODE_INDEX

        siblings = self._fetch(self.llt)

        for i, ns in enumerate(siblings):
            if ns.code > 0:
                siblings[i].sub_key = self.llt.sub_key + [ns.code - ROOT_NODE_BASE]

        _ = self._insert(siblings)

        return self.trie, self.llt

    def _resize(self, size):

        self.trie.base = np.append(self.trie.base, np.zeros(size - len(self.trie.base), int))
        self.trie.check = np.append(self.trie.check, np.zeros(size - len(self.trie.check), int))
        self.used = np.append(self.used, np.zeros(size - len(self.used), bool))

    def _fetch(self, parent: State):

        prev = 0
        siblings = []

        for i in range(parent.left, parent.right):

            if len(self.key[i]) < parent.depth:
                continue

            tmp = self.key[i]
            cur = 0
            if len(self.key[i]) != parent.depth:
                cur = ord(tmp[parent.depth]) + 1

            if prev > cur:
                raise ValueError()

            if cur != prev or len(siblings) == 0:

                if cur != 0:
                    sub_key = parent.sub_key + [cur - ROOT_NODE_BASE]
                else:
                    sub_key = parent.sub_key

                tmp_node = State()
                tmp_node.depth = parent.depth + 1
                tmp_node.code = cur
                tmp_node.left = i
                tmp_node.sub_key = sub_key

                if len(siblings) != 0:
                    siblings[len(siblings) - 1].right = i

                siblings.append(tmp_node)
                if len(parent.children) != 0:
                    parent.children[len(parent.children) - 1].right = i
                parent.children.append(tmp_node)

            prev = cur

        if len(siblings) != 0:
            siblings[len(siblings) - 1].right = parent.right
        if len(parent.children) != 0:
            parent.children[len(siblings) - 1].right = parent.right

        return parent.children

    def _insert(self, siblings):
        begin = 0
        pos = max(siblings[0].code + 1, self.next_check_pos) - 1
        non_zero_num = 0
        first = False

        if len(self.trie.base) <= pos:
            self._resize(pos + 1)

        while True:
            pos += 1

            if len(self.trie.base) <= pos:
                self._resize(pos + 1)

            if self.trie.check[pos] > 0:
                non_zero_num += 1
                continue
            elif not first:
                self.next_check_pos = pos
                first = True

            begin = pos - siblings[0].code
            if len(self.trie.base) <= (begin + siblings[-1].code):
                self._resize(begin + siblings[-1].code + RESIZE_DELTA)

            if self.used[begin]:
                continue

            flag = True
            for i in range(1, len(siblings)):
                if self.trie.check[begin + int(siblings[i].code)] != 0:
                    flag = False
                    break

            if flag:
                break

        if non_zero_num / (pos - self.next_check_pos + 1) >= 0.95:
            self.next_check_pos = pos

        self.used[begin] = True

        for sibling in siblings:
            self.trie.check[begin + sibling.code] = begin

        for sibling in siblings:
            new_siblings = self._fetch(sibling)

            if len(new_siblings) == 0:
                self.trie.base[begin + sibling.code] = -sibling.left - 1
                self.output[begin + sibling.code] = sibling.sub_key
                sibling.base = END_NODE_BASE
                sibling.index = begin + sibling.code
            else:
                h = self._insert(new_siblings)
                self.trie.base[begin + sibling.code] = h
                sibling.index = begin + sibling.code
                sibling.base = h

        return begin


class Machine:

    def __init__(self):
        self.trie = None
        self.failure = None
        self.output = None

    def serialize(self, path):

        self.trie.serialize(path)
        np.save(os.path.join(path, 'failure.npy'), self.failure)

        with open(os.path.join(path, 'output.pkl'), 'wb') as f:
            pickle.dump(self.output, f)

    def deserialize(self, path):

        self.trie.deserialize(path)

        self.failure = np.load(os.path.join(path, 'failure.npy'))

        with open(os.path.join(path, 'output.pkl'), 'rb') as f:
            self.output = pickle.load(f)

    def add_keywords_from_list(self, keywords):

        builder = TrieBuilder()
        self.trie, llt = builder.add_keywords_from_list(keywords)
        self.failure = np.zeros(len(self.trie.base), int)

        self.output = {}
        for idx, val in builder.output.items():
            if idx not in self.output:
                self.output[idx] = []
            self.output[idx].append(val)

        queue = []
        for c in llt.children:
            self.failure[c.base] = ROOT_NODE_BASE
        queue.extend(llt.children)

        while len(queue):

            node = queue[0]
            for n in node.children:
                if n.base == END_NODE_BASE:
                    continue
                in_state = self._f(node.base)

                while True:
                    out_state = self._g(in_state, n.code - ROOT_NODE_BASE)
                    if out_state == FAIL_STATE:
                        in_state = self._f(in_state)
                    else:
                        break

                if out_state in self.output:
                    copy_out_state = []
                    for o in self.output[out_state]:
                        copy_out_state.append(o)
                    self.output[n.base] = copy_out_state + self.output[n.base]

                self._set_f(n.base, out_state)

            queue.extend(node.children)
            queue = queue[1:]

    def _g(self, in_state, input):
        if in_state == FAIL_STATE:
            return ROOT_STATE

        t = in_state + input + ROOT_NODE_BASE
        if t >= len(self.trie.base):
            if in_state == ROOT_STATE:
                return ROOT_STATE
            return FAIL_STATE
        if in_state == self.trie.check[t]:
            return self.trie.base[t]
        if in_state == ROOT_STATE:
            return ROOT_STATE

        return FAIL_STATE

    def _f(self, index):
        return self.failure[index]

    def _set_f(self, in_state, out_state):
        self.failure[in_state] = out_state

    def __getitem__(self, sentence):
        state = ROOT_STATE
        ans = []
        for pos, c in enumerate(sentence):
            c = ord(c)
            if self._g(state, c) == FAIL_STATE:
                state = self._f(state)
            else:
                state = self._g(state, c)
                if state in self.output:
                    val = self.output[state]
                    for keyword in val:
                        term = ''
                        for word in keyword:
                            term += chr(word)

                        ans.append((term, pos - len(term) + 1, pos + 1))

        return ans
