
from typing import TypeVar
from queue import Queue


T = TypeVar('T')
class HashQueue():
    """
    This data structure uses a hash table to achieve:
    - O(1) inserts, O(1) finds. -- Using a hashmap.
    - O(1) remove first element, and O(1) find first element. -- using a queue.
    """
    def __init__(self, items: list[T] = []):
        self.hashmap: dict[T, bool] = {item: True for item in items}
        self.queue: Queue[T] = Queue()
        for item in items:
            self.queue.put(item)

    def __str__(self) -> str:
        map_to_list = list(self.hashmap.keys())
        return f"head: {self.peek()}, items: {list(self.queue.queue)}"
        
    def add(self, item):
        # Already exists
        if (self.hashmap.get(item) != None):
            return # Already added, but fail silently.
        
        # Add to hashmap and queue
        self.hashmap[item] = True
        self.queue.put(item)
        pass

    def has(self, item: T):
        exists = self.hashmap.get(item)

        if exists is not None:
            return True
        return False

    def pop(self):
        # Already empty
        if self.queue.empty():
            return None
        
        # Remove from hashmap and queue
        item = self.queue.get()
        self.hashmap.pop(item)
        return item

    def peek(self):
        if self.queue.empty():
            return None
        else:
            return self.queue.queue[0]


# UNIT TEST
# self.recentlyDeletedHashQueue = HashQueue(['FAT32'])
# print(self.recentlyDeletedHashQueue, "__str__ should be 'head: FAT32, items: ['FAT32]'")
# print(self.recentlyDeletedHashQueue.has('FAT32'), "- .has('FAT32') should be True")
# print(self.recentlyDeletedHashQueue.has('FAT322'), "- .has('FAT322') should be False")
# print(self.recentlyDeletedHashQueue.add('FAT322'), "- .add('FAT322') should be None")
# print(self.recentlyDeletedHashQueue.has('FAT322'), "- .has('FAT322') should be True")
# print(self.recentlyDeletedHashQueue.pop(), "- .pop() should be FAT32")
# print(self.recentlyDeletedHashQueue.peek(), "- .peek() should FAT322")
# print(self.recentlyDeletedHashQueue.pop(), "- .pop() should FAT322")
# print(self.recentlyDeletedHashQueue.peek(), "- .peek() should None")