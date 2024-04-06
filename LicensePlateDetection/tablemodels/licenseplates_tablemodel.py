from typing import List
from PySide6 import QtCore

class LicensePlatesTableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent, initialList, header, *args):
        QtCore.QAbstractTableModel.__init__(self, parent, *args)
        self.mylist: List[tuple[str, str]] = initialList
        self.header = header

    def rowCount(self, parent):
        return len(self.mylist)

    def columnCount(self, parent):
        return len(self.header)

    def data(self, index, role):
        if not index.isValid():
            return None
        elif role != QtCore.Qt.DisplayRole:
            return None
        return self.mylist[index.row()][index.column()]

    def insertRows(self, position, rows, index=QtCore.QModelIndex()):
        self.beginInsertRows(QtCore.QModelIndex(), position, position + len(rows) - 1)
        for index, row in enumerate(rows):
            self.mylist.insert(position + index, row)
        self.endInsertRows()
    
    def addRows(self, rows: List[tuple[str, str]]):
        first = len(self.mylist) - 1
        last = first + len(rows) - 1
        
        self.beginInsertRows(QtCore.QModelIndex(), first, last)
        for item in rows:
            self.mylist.append(item)
        self.endInsertRows()

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.header[col]
        return None
    
    

    # def sort(self, col, order):
    #     """sort table by given column number col"""
    #     self.emit(QtCore.SIGNAL("layoutAboutToBeChanged()"))
    #     self.mylist = sorted(self.mylist,
    #                          key=operator.itemgetter(col))
    #     if order == QtCore.Qt.DescendingOrder:
    #         self.mylist.reverse()
    #     self.emit(QtCore.SIGNAL("layoutChanged()"))
