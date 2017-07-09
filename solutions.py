import unittest
import Queue
from collections import deque
import sys


class CharMap:
    """Index word by letters."""

    def __init__(self, word, supressIndices=False):
        """Initialize char map."""
        self.supressIndices = supressIndices
        self.charIndices = {}
        self.__loadIndices__(word)

    def __loadIndices__(self, word):
        """
        Load indices from the word.

        Creates a dictonary from a word where
        each letter is the key and the value is
        a list of its indices or the number of letter of each indice
        depending on the status of supressIndices field.

        For exemple,
        Given the word 'test'
        Our dictonary should store the following structure:
         if supressIndices is False
            {
              't': [0, 3],
              'e': [1],
              's': [2]
            }
         if supressIndices is True
            {
              't': 2,
              'e': 1,
              's': 1
            }
        """
        def withIndices(letter, index):
            if letter in self.charIndices:
                self.charIndices[letter].append(index)
            else:
                self.charIndices[letter] = [index]

        def withoutIndices(letter, index):
            hasKey = letter in self.charIndices
            cnt = self.charIndices[letter] + 1 if hasKey else 1
            self.charIndices[letter] = cnt

        storeLetters = withoutIndices if self.supressIndices else withIndices

        for i in range(len(word)):
            letter = word[i]
            storeLetters(letter, i)

    def count(self, letter):
        """Return number of indices for a given letter."""
        values = self.values(letter)
        return values if self.supressIndices else len(values)

    def values(self, letter):
        """Return the indices for a given letter."""
        return self.charIndices[letter]

    def hasLetter(self, letter):
        """
        Check if a letter exists in the charIndices.

        Return True if the charIndices dictonary contains the letter,
        otherwise returns False.
        """
        return letter in self.charIndices


class Node(object):
    def __init__(self, data):
        self.data = data
        self.next = None

    def __repr__(self):
        return '(%s)' % (self.data)


def question1(s, t):
    """
    Question 1.

    Given two strings s and t, determine whether some anagram of t
    is a substring of s.

    For example: if s = "udacity" and t = "ad", then the function
    returns True.

    Your function definition should look like: question1(s, t) and
    return a boolean True or False.
    """
    if s and t:
        map1 = CharMap(s, True)
        map2 = CharMap(t, True)

        for k in map2.charIndices:
            if not map1.hasLetter(k) or map1.count(k) < map2.count(k):
                return False

        return True

    return False


def question2(a):
    """
    Question 2.

    Given a string a, find the longest palindromic substring contained in a.

    Your function definition should look like question2(a), and return a
    string.
    """
    if not a:
        return ""

    class Word:

        def __init__(self, lowerBound, upperBound, letter):
            self.lowerBound = lowerBound
            self.upperBound = upperBound
            self.priority = upperBound - lowerBound + 1
            self.letter = letter
            self.word = None

        def getWord(self, completeWord):
            if not self.word:
                self.word = completeWord[self.lowerBound: self.upperBound + 1]

            return self.word

        def isPalindrome(self, completeWord):
            """
            Check if the word is a palindrome.

            Return True if is a palindrome and False if it's not.
            """
            word = self.getWord(completeWord)
            lenWord = len(word)

            for index in range(lenWord / 2):
                if word[index] != word[lenWord - index - 1]:
                    return False

            return True

        def __repr__(self):
            return '(%s, %s, %s, %s)' % (self.lowerBound,
                                         self.upperBound,
                                         self.priority,
                                         self.letter)

        def __cmp__(self, other):
            return other.priority - self.priority

    class WordGroups:

        def __init__(self, word):
            """Initialize pair groups."""
            self.charMap = CharMap(word)
            self.indexGroupSize = {}
            self.numWordPerLetterPair = {}
            self.words = Queue.PriorityQueue()
            self.__setGroupsSize__()

            for k in self.charMap.charIndices:
                self.__calculateWordsFromLetterPair__(k)

        def __setGroupsSize__(self):
            for k in self.charMap.charIndices:
                self.indexGroupSize[k] = self.charMap.count(k)

        def __decrementGroupSize__(self, letter):
            size = self.indexGroupSize[letter]
            self.indexGroupSize[letter] = size - 1

            numberOfWords = self.charMap.count(letter) - size + 1
            self.numWordPerLetterPair[letter] = numberOfWords

            return size

        def __calculateWordsFromLetterPair__(self, letter):
            newSize = self.__decrementGroupSize__(letter)
            values = self.charMap.values(letter)

            if newSize >= 2:
                for i in range(self.charMap.count(letter) - newSize + 1):
                    lowerBound = values[i]
                    upperBound = values[i + newSize - 1]
                    self.words.put(Word(lowerBound, upperBound, letter))
            else:
                self.words.put(Word(values[0], values[0], letter))

        def popWord(self):
            word = self.words.get()
            self.numWordPerLetterPair[word.letter] -= 1
            return word

        def hasMoreWords(self):
            return not self.words.empty()

        def refreshWordList(self, letter):
            if self.numWordPerLetterPair[letter] == 0:
                self.__calculateWordsFromLetterPair__(letter)

    groups = WordGroups(a)

    while groups.hasMoreWords():
        newWord = groups.popWord()

        if newWord.isPalindrome(a):
            return newWord.getWord(a)

        groups.refreshWordList(newWord.letter)

    return ""


def question3(G):
    """
    Question 3.

    Given an undirected graph G, find the minimum spanning tree within G.
    A minimum spanning tree connects all vertices in a graph with the
    smallest possible total weight of edges.

    Your function should take in and return an adjacency list structured
    like this:

        {
            'A': [('B', 2)],
            'B': [('A', 2), ('C', 5)],
            'C': [('B', 5)]
        }

    Vertices are represented as unique strings.

    The function definition should be question3(G)

    External resources:
        - https://en.wikipedia.org/wiki/Minimum_spanning_tree;
        - https://en.wikipedia.org/wiki/Prim%27s_algorithm;
        - https://www.youtube.com/watch?v=z1L3rMzG1_A.
    """
    if not (G and len(G)):
        return None

    parents = {}
    weightPerVertex = {}

    def addElement(dict, key, value):
        list = dict[key] if key in dict else []
        list.append(value)
        dict[key] = list

    def findMinWeight():
        minValue = min(weightPerVertex.values())

        for vertex in weightPerVertex:
            if weightPerVertex[vertex] == minValue:
                Q.remove(vertex)
                del weightPerVertex[vertex]
                return vertex, minValue

    def visitNextVertex(otherVertex, weight):
        hasKey = otherVertex in weightPerVertex
        if hasKey and weightPerVertex[otherVertex] > weight:
            parents[otherVertex] = vertex
            weightPerVertex[otherVertex] = weight

    F = {}
    Q = set(G.keys())

    for vertex in G:
        weightPerVertex[vertex] = sys.maxint
        parents[vertex] = None

    for vertex in G:
        weightPerVertex[vertex] = 0
        break

    while len(Q) > 0:
        vertex, weight = findMinWeight()

        if parents[vertex] is not None:
            parent = parents[vertex]

            addElement(F, parent, (vertex, weight))
            addElement(F, vertex, (parent, weight))

        for edge in G[vertex]:
            visitNextVertex(edge[0], edge[1])

    return F


def question4(T, r, n1, n2):
    """
    Question 4.

    Find the least common ancestor between two nodes on a binary search
    tree. The least common ancestor is the farthest node from the root that
    is an ancestor of both nodes. For example, the root is a common
    ancestor of all nodes on the tree, but if both nodes are descendents of
    the root's left child, then that left child might be the lowest common
    ancestor. You can assume that both nodes are in the tree, and the tree
    itself adheres to all BST properties.

    The function definition should look like question4(T, r, n1, n2),
    where T is the tree represented as a matrix, where the index of the
    list is equal to the integer stored in that node and a 1 represents a
    child node, r is a non-negative integer representing the root, and n1
    and n2 are non-negative integers representing the two nodes in no
    particular order.

    For example, one test case might be

        question4([[0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0]],
                  3,
                  1,
                  4)

    and the answer would be 3.

    """
    if T is None or r is None or n1 is None or n2 is None:
        return None

    if r < 0 or n1 < 0 or n2 < 0:
        return None

    if len(T) < max(n1, n2, r) or len(T[0]) < max(n1, n2, r):
        return None

    availableNodes = range(len(T))

    def findParent(column):
        for row in availableNodes:
            if T[row][column] == 1:
                return row

        raise Exception("Node %s does not make part of the tree" % column)

    def findAncestors(startNode, stopCallBack, unordered=False):
        """Find a path to root or if stopCallBack is evalute to True."""
        path = set() if unordered else []
        parent = startNode

        while parent != r:

            parent = findParent(parent)
            if unordered:
                path.add(parent)
            else:
                path.append(parent)

            # stop when find a common node
            if stopCallBack and stopCallBack(parent):
                break

        return path

    def findFarthestNode():
        ancestorN1 = findAncestors(n1, None, True)
        ancestorN2 = findAncestors(n2, lambda node: node in ancestorN1)

        return ancestorN2.pop()

    return findFarthestNode()


def question5(ll, m):
    """
    Question 5.

    Find the element in a singly linked list that's m elements from the end.

    For example, if a linked list has 5 elements, the 3rd element from the
    end is the 3rd element. The function definition should look like
    question5(ll, m), where ll is the first node of a linked list and m is
    the "mth number from the end".

    You should copy/paste the Node class below to use as a representation
    of a node in the linked list.

    Return the value of the node at that position.

        class Node(object):
          def __init__(self, data):
            self.data = data
            self.next = None
    """
    if not (ll and m):
        return None

    dq = deque(maxlen=m)
    nextNode = ll

    while nextNode:
        dq.append(nextNode)
        nextNode = nextNode.next

    return dq.popleft().data if len(dq) >= m else None


class TestQuestions(unittest.TestCase):

    def test_question1_edge_cases(self):
        """Test question 1 - edge cases."""
        self.assertFalse(question1(None, "ad"))
        self.assertFalse(question1("udacity", None))
        self.assertFalse(question1("udacity", ""))
        self.assertFalse(question1("", "ad"))
        self.assertFalse(question1("", ""))

    def test_question1_should_pass(self):
        """Test question 1 - positive cases."""
        self.assertTrue(question1("udacity", "ad"))
        self.assertTrue(question1("u", "u"))
        self.assertTrue(question1("uaddda", "aadddu"))

    def test_question1_should_fail(self):
        """Test question 1 - negative cases."""
        self.assertFalse(question1("udacity", "adx"))
        self.assertFalse(question1("u", "ud"))
        self.assertFalse(question1("uaddda", "aaddddu"))

    def test_question2_edge_cases(self):
        """Test question 2 - edge cases."""
        self.assertEquals(question2(None), "")
        self.assertEquals(question2(""), "")

    def test_question2_should_pass(self):
        """Test question 2."""
        self.assertEquals(question2("abbcccbba"), "abbcccbba")
        self.assertEquals(question2("abbccbba"), "abbccbba")
        self.assertEquals(question2("abbcba"), "bcb")
        self.assertEquals(len(question2("xabcdefax")), 1)
        self.assertEquals(question2("a"), "a")

    def test_question3_edge_cases(self):
        """Test question 3 - edge cases."""
        self.assertEquals(question3(None), None)
        self.assertEquals(question3({}), None)

    def test_question3_should_pass(self):
        """Test question 3."""
        A = {
            'A': [('B', 2)],
            'B': [('A', 2), ('C', 5)],
            'C': [('B', 5)]
        }

        B = {
            'A': [('B', 7), ('D', 5)],
            'B': [('A', 7), ('C', 8), ('D', 9), ('E', 7)],
            'C': [('B', 8), ('E', 5)],
            'D': [('A', 5), ('B', 9), ('E', 15), ('F', 6)],
            'E': [('B', 7), ('C', 5), ('D', 15), ('F', 8), ('G', 9)],
            'F': [('D', 6), ('E', 8), ('G', 11)],
            'G': [('E', 9), ('F', 11)]
        }

        C = {
            'A': [('B', 7), ('D', 5)],
            'B': [('A', 7), ('E', 7)],
            'C': [('E', 5)],
            'D': [('A', 5), ('F', 6)],
            'E': [('B', 7), ('C', 5), ('G', 9)],
            'F': [('D', 6)],
            'G': [('E', 9)]
        }

        def sortResult(Q):
            for q in Q:
                Q[q].sort()
            return Q

        self.assertEquals(question3(A), A)
        self.assertEquals(sortResult(question3(B)), C)

    def test_question4_edge_case(self):
        """Test question 4."""
        self.assertEquals(question4(None,
                                    3,
                                    1,
                                    4), None)
        self.assertEquals(question4([[]],
                                    3,
                                    1,
                                    4), None)
        self.assertEquals(question4([[0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0]],
                                    None,
                                    1,
                                    4), None)
        self.assertEquals(question4([[0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0]],
                                    3,
                                    None,
                                    4), None)
        self.assertEquals(question4([[0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0]],
                                    3,
                                    1,
                                    None), None)

    def test_question4(self):
        """Test question 4."""
        self.assertEquals(question4([[0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0]],
                                    3,
                                    1,
                                    4), 3)
        self.assertEquals(question4([[0, 1, 1, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0]],
                                    3,
                                    1,
                                    2), 0)
        self.assertEquals(question4([[0, 1, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0, 0, 0]],
                                    0,
                                    5,
                                    6), 4)

    def test_question5_edge_cases(self):
        """Test question 5 - edge cases."""
        self.assertEquals(question5(None, 3), None)
        self.assertEquals(question5(Node('A'), None), None)

    def test_question5(self):
        """Test question 5."""
        head = Node('A')
        head.next = Node('B')
        head.next.next = Node('C')
        head.next.next.next = Node('D')
        head.next.next.next.next = Node('E')
        self.assertEquals(question5(head, 3), 'C')
        self.assertEquals(question5(head, 2), 'D')
        head.next.next.next.next.next = Node('F')
        self.assertEquals(question5(head, 2), 'E')
        self.assertEquals(question5(head, 6), 'A')
        self.assertEquals(question5(head, 1), 'F')
        self.assertEquals(question5(head, 10), None)


if __name__ == '__main__':
    unittest.main()
