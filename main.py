
class Node:
    def __init__(self, data=None, next=None):
        self.data = data
        self.next = next


class LinkedList:
    def __init__(self):
        self.head = None

    def insert_to_begining(self, data):
        node = Node(data, self.head)
        self.head = node

    def insert_to_end(self, data):
        if self.head is None:
            self.head = Node(data, None)
            return

        itr = self.head

        while itr.next:
            itr = itr.next

        itr.next = Node(data, None)

    def insert_values(self, data_list):
        self.head = None
        for data in data_list:
            self.insert_to_end(data)

    def print(self):
        if self.head is None:
            print("Empty linked list")
            return

        itr = self.head
        llstr = ''

        while itr:
            llstr += str(itr.data)+ '--->'
            itr = itr.next

        print(llstr)



if __name__ == "__main__":
    ll = LinkedList()
    ll.insert_values(["nana", "gpt", "gemini", "claude"])
    ll.print()
