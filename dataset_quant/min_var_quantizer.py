

class KDTree:
    def __init__(self, root, left = None, right = None):
        self.root = root
        self.left = left
        self.right = right
    
    def set_left(self, left):
        self.left = left
    
    def set_right(self, right):
        self.right = right

    def count_leaves(self):
        # print(self.root)
        if self.root is None:
            return 0
        if self.left is None and self.right is None:
            # print(self.root.thresh)
            return 1
        left = self.left.count_leaves()
        right = self.right.count_leaves()
        return left + right

class Partition:
    def __init__(self,img):
        self.image = img
        self.left = None
        self.right = None
        self.thresh = None
        self.priority = None

    def find_optimal_split(self):
        var = np.var(self.image)

        # split pixels according to partition otsu
        self.thresh = filters.threshold_otsu(self.image)
        self.left = np.asarray([i for i in self.image if  i <= self.thresh])
        self.right = np.asarray([i for i in self.image if i > self.thresh])

        left_var = np.var(self.left)
        right_var = np.var(self.right)
        self.priority = var - (left_var + right_var)

    def split(self):
        return Partition(self.left), Partition(self.right)