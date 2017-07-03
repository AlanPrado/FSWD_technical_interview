import unittest


class CharMap:
    """Index word by letters."""

    def __init__(self, word):
        """Initialize char map."""
        self.charIndices = {}
        self.__loadIndices__(word)

    def __loadIndices__(self, word):
        """
        Load indices from the word.

        Creates a dictonary from a word where
        each letter is the key and the value is
        a list of its indices.

        For exemple,
        Given the word 'test'
        Our dictonary should store the following structure
        {
          't': [0, 3],
          'e': [1],
          's': [2]
        }
        """
        for i in range(len(word)):
            letter = word[i]

            if letter in self.charIndices:
                self.charIndices[letter].append(i)
            else:
                self.charIndices[letter] = [i]

    def count(self, letter):
        """Return number of indices for a given letter."""
        return len(self.charIndices[letter])

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
        map1 = CharMap(s)
        map2 = CharMap(t)

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

    class WordGroups:

        def __init__(self, word):
            """Initialize pair groups."""
            self.charMap = CharMap(word)
            self.indexGroupSize = {}
            self.numWordPerLetterPair = {}
            self.words = []
            self.__setGroupsSize__()

            for k in self.charMap.charIndices:
                self.__pushWordsPairLetterPair__(k)

            self.words.sort(self.__sortByPriority__)

        def __setGroupsSize__(self):
            for k in self.charMap.charIndices:
                self.indexGroupSize[k] = self.charMap.count(k)

        def __decrementGroupSize__(self, letter):
            size = self.indexGroupSize[letter]
            self.indexGroupSize[letter] = size - 1

            numberOfWords = self.charMap.count(letter) - size + 1
            self.numWordPerLetterPair[letter] = numberOfWords

            return size

        def __sortByPriority__(self, word1, word2):
            return word1.priority - word2.priority

        def __pushWordsPairLetterPair__(self, letter):
            newSize = self.__decrementGroupSize__(letter)
            values = self.charMap.values(letter)

            if newSize >= 2:
                for i in range(self.charMap.count(letter) - newSize + 1):
                    lowerBound = values[i]
                    upperBound = values[i + newSize - 1]
                    self.words.append(Word(lowerBound, upperBound, letter))
            else:
                self.words.append(Word(values[0], values[0], letter))

        def popWord(self):
            word = self.words.pop()
            self.numWordPerLetterPair[word.letter] -= 1
            return word

        def hasMoreWords(self):
            return len(self.words) != 0

        def refreshWordList(self, letter):
            if self.numWordPerLetterPair[letter] == 0:
                self.__pushWordsPairLetterPair__(letter)
                self.words.sort(self.__sortByPriority__)

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
    """
    pass


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
    if not (T and r and n1 and n2):
        return None

    if r <= 0 or n1 <= 0 or n2 <= 0:
        return None

    if len(T) < max(n1, n2, r) or len(T[0]) < max(n1, n2, r):
        return None

    root = r - 1
    node1 = n1 - 1
    node2 = n2 - 1
    cacheParent = {}

    def findParent(column):
        if column in cacheParent:
            return cacheParent[column]

        for row in range(len(T)):
            if T[row][column] == 1:
                cacheParent[column] = row
                return row

        # if don't find any parent, the parent should be the root
        cacheParent[column] = root
        return root

    def findPathToRoot(startNode):
        path = []
        parent = startNode

        while (parent != root):
            parent = findParent(parent)
            path.append(parent)

        return path

    def findFarthestNode():
        pathN1 = findPathToRoot(node1)
        pathN2 = findPathToRoot(node2)

        for node in pathN1:
            if node in pathN2:
                return node

    return findFarthestNode() + 1


def question5(ll, m):
    """
    Question 5.

    Find the element in a singly linked list that's m elements from the
    end.

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

    list = []
    nextNode = ll

    while nextNode:
        list.append(nextNode)
        nextNode = nextNode.next

    return list[-m].data if len(list) > m else -1


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
        self.assertEquals(question2("xabcdefax"), "a")
        self.assertEquals(question2("a"), "a")

    def test_question3_should_pass(self):
        """Test question 3."""
        pass

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
        self.assertEquals(question4([[0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0]],
                                    3,
                                    1,
                                    5), 4)

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
        self.assertEquals(question5(head, 10), -1)


if __name__ == '__main__':
    unittest.main()
