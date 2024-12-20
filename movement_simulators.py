import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class PourMovementBFT:
    def __init__(self, image, start_loc, display_traversal=False):
        self.image = image
        self.output = np.zeros_like(image, dtype=bool)
        self.queue = [start_loc]
        self.display_traversal_settings(display_traversal)
        self.visualize_traversal = lambda: None
        if display_traversal:
            self.visualize_traversal = self._visualize_traversal

    def _visualize_traversal(self):
        fig, ax = plt.subplots()
        fig.axes[0].axis("off")
        image_temp = np.stack((self.image, self.image, self.image), axis=2)
        image_temp = (1 - image_temp).astype(np.int16) * (0, 0, 255)
        img_ax = ax.imshow(image_temp, vmin=0, vmax=1, origin="upper")
        def animate(i):
            image_temp[*self.visited[i]] = (255, 255, 0)
            img_ax.set_data(image_temp)
            return img_ax
        def no_animation(i):
            img_ax.set_data(image_temp)
            return img_ax
        if len(self.visited) == 0:
            anim = FuncAnimation(fig, no_animation, frames=1, interval=5)
        else:
            anim = FuncAnimation(fig, animate, frames=len(self.visited), interval=5)
        return anim

    def display_traversal_settings(self, display_traversal):
        if display_traversal:
            self.visited = []
            self.insert_visited_node = self.visited.append
        else:
            self.visited = set()
            self.insert_visited_node = self.visited.add

    def get_neighbors(self, node):
        """Function that returns the next nodes to visit"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def _get_neighbors(self, node, valid_directions):
        neighbors = []
        for dx, dy in valid_directions:
            neighbor = (node[0] + dx, node[1] + dy)
            if neighbor[0] >= self.image.shape[0] or neighbor[0] < 0 or neighbor[1] >= self.image.shape[1] or neighbor[1] < 0:
                continue
            if self.image[neighbor[0], neighbor[1]] == 1:
                continue
            neighbors.append(neighbor)
        return neighbors
    
    def traverse(self):
        initial_node = self.queue[0]
        if self.image[*initial_node] == 1:
            return self.visited
        self.insert_visited_node(initial_node)
        while len(self.queue) > 0:
            node = self.queue.pop(0)
            self.output[*node] = True
            for n in self.get_neighbors(node):
                if n not in self.visited:
                    self.insert_visited_node(n)
                    self.queue.append(n)
        return self.visited
    

class FullPour(PourMovementBFT):
    def __init__(self, image, display_traversal=False):
        start_loc = (0, 0)
        super().__init__(image, start_loc, display_traversal)
        
    def get_neighbors(self, node):
        valid_directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        return self._get_neighbors(node, valid_directions)
    

class TopPourLeftSide(PourMovementBFT):
    def __init__(self, image, display_traversal=False):
        image = image[:, :image.shape[1]//2]
        start_loc = (0, 0)
        super().__init__(image, start_loc, display_traversal)
    
    def get_neighbors(self, node):
        valid_directions = [(0, -1), (0, 1), (1, 0)]
        return self._get_neighbors(node, valid_directions)
    

class TopPourRightSide(PourMovementBFT):
    def __init__(self, image, display_traversal=False):
        image = image[:, image.shape[1]//2:]
        start_loc = (0, image.shape[1]-1)
        super().__init__(image, start_loc, display_traversal)
    
    def get_neighbors(self, node):
        valid_directions = [(0, -1), (0, 1), (1, 0)]
        return self._get_neighbors(node, valid_directions)
    

class BottomPourLeftSide(PourMovementBFT):
    def __init__(self, image, display_traversal=False):
        image = image[:, :image.shape[1]//2]
        start_loc = (image.shape[0]-1, 0)
        super().__init__(image, start_loc, display_traversal)
    
    def get_neighbors(self, node):
        valid_directions = [(0, -1), (0, 1), (-1, 0)]
        return self._get_neighbors(node, valid_directions)
    
    
class BottomPourRightSide(PourMovementBFT):
    def __init__(self, image, display_traversal=False):
        image = image[:, image.shape[1]//2:]
        start_loc = (image.shape[0]-1, image.shape[1]-1)
        super().__init__(image, start_loc, display_traversal)
    
    def get_neighbors(self, node):
        valid_directions = [(0, -1), (0, 1), (-1, 0)]
        return self._get_neighbors(node, valid_directions)
    
    
class LeftPourTopSide(PourMovementBFT):
    def __init__(self, image, display_traversal=False):
        image = image[:image.shape[0]//2, :]
        start_loc = (0, 0)
        super().__init__(image, start_loc, display_traversal)
    
    def get_neighbors(self, node):
        valid_directions = [(0, -1), (0, 1), (1, 0)]
        return self._get_neighbors(node, valid_directions)
    

class LeftPourBottomSide(PourMovementBFT):
    def __init__(self, image, display_traversal=False):
        image = image[:image.shape[0]//2, :]
        start_loc = (image.shape[0]-1, 0)
        super().__init__(image, start_loc, display_traversal)
    
    def get_neighbors(self, node):
        valid_directions = [(0, -1), (0, 1), (-1, 0)]
        return self._get_neighbors(node, valid_directions)


class RightPourTopSide(PourMovementBFT):
    def __init__(self, image, display_traversal=False):
        image = image[image.shape[0]//2:, :]
        start_loc = (0, image.shape[1]-1)
        super().__init__(image, start_loc, display_traversal)
    
    def get_neighbors(self, node):
        valid_directions = [(0, -1), (0, 1), (1, 0)]
        return self._get_neighbors(node, valid_directions)
    

class RightPourBottomSide(PourMovementBFT):
    def __init__(self, image, display_traversal=False):
        image = image[image.shape[0]//2:, :]
        start_loc = (image.shape[0]-1, image.shape[1]-1)
        super().__init__(image, start_loc, display_traversal)
    
    def get_neighbors(self, node):
        valid_directions = [(0, -1), (0, 1), (-1, 0)]
        return self._get_neighbors(node, valid_directions)


class ExecuteAllPourMovements:
    def __init__(self, display_traversal=False):
        self.display_traversal = display_traversal
        
    def get_objects(self, image):
        # TopPourLeftSide
        top_pour_left = TopPourLeftSide(image, self.display_traversal)
        # TopPourRightSide
        top_pour_right = TopPourRightSide(image, self.display_traversal)
        # BottomPourLeftSide
        bottom_pour_left = BottomPourLeftSide(image, self.display_traversal)
        # BottomPourRightSide
        bottom_pour_right = BottomPourRightSide(image, self.display_traversal)
        # LeftPourTopSide
        left_pour_top = LeftPourTopSide(image, self.display_traversal)
        # LeftPourBottomSide
        left_pour_bottom = LeftPourBottomSide(image, self.display_traversal)
        # RightPourTopSide
        right_pour_top = RightPourTopSide(image, self.display_traversal)
        # RightPourBottomSide
        right_pour_bottom = RightPourBottomSide(image, self.display_traversal)
        # FullPour
        full_pour = FullPour(image, self.display_traversal)

        for pour in [top_pour_left, top_pour_right, bottom_pour_left, bottom_pour_right, left_pour_top, left_pour_bottom, right_pour_top, right_pour_bottom, full_pour]:
            pour.traverse()
        
        return top_pour_left, top_pour_right, bottom_pour_left, bottom_pour_right, left_pour_top, left_pour_bottom, right_pour_top, right_pour_bottom, full_pour

if __name__ == "__main__":
    import numpy as np
    img = np.zeros((10, 10), dtype=bool)
    img[5, 1:9] = True
    img[1:9, 5] = True
    img[3:9, 1] = True
    img[2, 3:7] = True
    img[0, 0] = True
    img[0, 6:9] = True
    # img[0, :] = True
    # img[:, 0] = True
    # img[-1, :] = True
    # img[:, -1] = True
    img = np.pad(img, pad_width=1, mode="constant", constant_values=False)
    bft = BottomPourLeftSide(img, display_traversal=True)
    plt.imshow(img, vmin=0, vmax=1, origin="upper")
    print(bft.traverse())
    # print(img)
    # print(bft.output)
    bft.visualize_traversal()

