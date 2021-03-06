


import gzip
import os
import platform
import sys
from PyQt5.QtCore import (QAbstractTableModel, QDateTime, QModelIndex, QSize, QTimer, QVariant, Qt,pyqtSignal)
from PyQt5.QtGui import ( QColor, QCursor, QFont, QFontDatabase, QFontMetrics, QPainter, QPalette, QPixmap)
from PyQt5.QtWidgets import QApplication,QDialog,QHBoxLayout, QLabel, QMessageBox,QScrollArea, QSplitter, QTableView,QWidget


(TIMESTAMP, TEMPERATURE, INLETFLOW, TURBIDITY, CONDUCTIVITY,
 COAGULATION, RAWPH, FLOCCULATEDPH) = range(8)

TIMESTAMPFORMAT = "yyyy-MM-dd hh:mm"


class WaterQualityModel(QAbstractTableModel):

    def __init__(self, filename):
        super(WaterQualityModel, self).__init__()
        self.filename = filename
        self.results = []


    def load(self):
        self.beginResetModel()
        exception = None
        fh = None
        try:
            if not self.filename:
                raise IOError("no filename specified for loading")
            self.results = []
            line_data = gzip.open(self.filename).read()
            for line in line_data.decode("utf8").splitlines():
                parts = line.rstrip().split(",")
                date = QDateTime.fromString(parts[0] + ":00",
                                            Qt.ISODate)

                result = [date]
                for part in parts[1:]:
                    result.append(float(part))
                self.results.append(result)

        except (IOError, ValueError) as e:
            exception = e
        finally:
            if fh is not None:
                fh.close()
            self.endResetModel()
            if exception is not None:
                raise exception


    def data(self, index, role=Qt.DisplayRole):
        if (not index.isValid() or
            not (0 <= index.row() < len(self.results))):
            return QVariant()
        column = index.column()
        result = self.results[index.row()]
        if role == Qt.DisplayRole:
            item = result[column]
            if column == TIMESTAMP:
                #item = item.toString(TIMESTAMPFORMAT)
                item=item
            else:
                #item = QString("%1").arg(item, 0, "f", 2)
                item = "{0:.2f}".format(item)
            return item
        elif role == Qt.TextAlignmentRole:
            if column != TIMESTAMP:
                return QVariant(int(Qt.AlignRight|Qt.AlignVCenter))
            return QVariant(int(Qt.AlignLeft|Qt.AlignVCenter))
        elif role == Qt.TextColorRole and column == INLETFLOW:
            if result[column] < 0:
                return QVariant(QColor(Qt.red))
        elif (role == Qt.TextColorRole and
              column in (RAWPH, FLOCCULATEDPH)):
            ph = result[column]
            if ph < 7:
                return QVariant(QColor(Qt.red))
            elif ph >= 8:
                return QVariant(QColor(Qt.blue))
            else:
                return QVariant(QColor(Qt.darkGreen))
        return QVariant()


    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.TextAlignmentRole:
            if orientation == Qt.Horizontal:
                return QVariant(int(Qt.AlignCenter))
            return QVariant(int(Qt.AlignRight|Qt.AlignVCenter))
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            if section == TIMESTAMP:
                return "Timestamp"
            elif section == TEMPERATURE:
                return "\u00B0" +"C"
            elif section == INLETFLOW:
                return "Inflow"
            elif section == TURBIDITY:
                return "NTU"
            elif section == CONDUCTIVITY:
                return "\u03BCS/cm"
            elif section == COAGULATION:
                return "mg/L"
            elif section == RAWPH:
                return "Raw Ph"
            elif section == FLOCCULATEDPH:
                return "Floc Ph"
        return int(section + 1)


    def rowCount(self, index=QModelIndex()):
        return len(self.results)


    def columnCount(self, index=QModelIndex()):
        return 8


class WaterQualityView(QWidget):
    clicked = pyqtSignal(QModelIndex)
    FLOWCHARS = (chr(0x21DC), chr(0x21DD), chr(0x21C9))

    def __init__(self, parent=None):
        super(WaterQualityView, self).__init__(parent)
        self.scrollarea = None
        self.model = None
        self.setFocusPolicy(Qt.StrongFocus)
        self.selectedRow = -1
        self.flowfont = self.font()
        size = self.font().pointSize()
        if platform.system() == "Windows":
            fontDb = QFontDatabase()
            for face in [face.lower() for face in fontDb.families()]:
                if face.find("unicode"):
                    self.flowfont = QFont(face, size)
                    break
            else:
                self.flowfont = QFont("symbol", size)
                WaterQualityView.FLOWCHARS = (chr(0xAC), chr(0xAE),
                                              chr(0xDE))


    def setModel(self, model):
        self.model = model
        #self.connect(self.model,
        #        SIGNAL("dataChanged(QModelIndex,QModelIndex)"),
        #        self.setNewSize)
        self.model.dataChanged.connect(self.setNewSize)
        #self.connect(self.model, SIGNAL("modelReset()"), self.setNewSize)
        self.model.modelReset.connect(self.setNewSize)
        self.setNewSize()


    def setNewSize(self):
        self.resize(self.sizeHint())
        self.update()
        self.updateGeometry()


    def minimumSizeHint(self):
        size = self.sizeHint()
        fm = QFontMetrics(self.font())
        size.setHeight(fm.height() * 3)
        return size


    def sizeHint(self):
        fm = QFontMetrics(self.font())
        size = fm.height()
        return QSize(fm.width("9999-99-99 99:99 ") + (size * 4),
                     (size / 4) + (size * self.model.rowCount()))


    def paintEvent(self, event):
        if self.model is None:
            return
        fm = QFontMetrics(self.font())
        timestampWidth = fm.width("9999-99-99 99:99 ")
        size = fm.height()
        indicatorSize = int(size * 0.8)
        offset = int(1.5 * (size - indicatorSize))
        minY = event.rect().y()
        maxY = minY + event.rect().height() + size
        minY -= size
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        y = 0
        for row in range(self.model.rowCount()):
            x = 0
            if minY <= y <= maxY:
                painter.save()
                painter.setPen(self.palette().color(QPalette.Text))
                if row == self.selectedRow:
                    painter.fillRect(x, y + (offset * 0.8),
                            self.width(), size, self.palette().highlight())
                    painter.setPen(self.palette().color(
                            QPalette.HighlightedText))
                #timestamp = self.model.data(
                        #self.model.index(row, TIMESTAMP)).toDateTime()
                timestamp = self.model.data(self.model.index(row, TIMESTAMP))
                painter.drawText(x, y + size,
                        timestamp.toString(TIMESTAMPFORMAT))
                #print(timestamp.toString(TIMESTAMPFORMAT))
                x += timestampWidth
                temperature = self.model.data(
                        self.model.index(row, TEMPERATURE))
                #temperature = temperature.toDouble()[0]
                temperature = float(temperature)
                if temperature < 20:
                    color = QColor(0, 0,
                            int(255 * (20 - temperature) / 20))
                elif temperature > 25:
                    color = QColor(int(255 * temperature / 100), 0, 0)
                else:
                    color = QColor(0, int(255 * temperature / 100), 0)
                painter.setPen(Qt.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(x, y + offset, indicatorSize,
                                    indicatorSize)
                x += size
                rawPh = self.model.data(self.model.index(row, RAWPH))
                #rawPh = rawPh.toDouble()[0]
                rawPh = float(rawPh)
                if rawPh < 7:
                    color = QColor(int(255 * rawPh / 10), 0, 0)
                elif rawPh >= 8:
                    color = QColor(0, 0, int(255 * rawPh / 10))
                else:
                    color = QColor(0, int(255 * rawPh / 10), 0)
                painter.setBrush(color)
                painter.drawEllipse(x, y + offset, indicatorSize,
                                    indicatorSize)
                x += size
                flocPh = self.model.data(
                        self.model.index(row, FLOCCULATEDPH))
                #flocPh = flocPh.toDouble()[0]
                flocPh = float(flocPh)
                if flocPh < 7:
                    color = QColor(int(255 * flocPh / 10), 0, 0)
                elif flocPh >= 8:
                    color = QColor(0, 0, int(255 * flocPh / 10))
                else:
                    color = QColor(0, int(255 * flocPh / 10), 0)
                painter.setBrush(color)
                painter.drawEllipse(x, y + offset, indicatorSize,
                                    indicatorSize)
                painter.restore()
                painter.save()
                x += size
                flow = self.model.data(
                        self.model.index(row, INLETFLOW))
                #flow = flow.toDouble()[0]
                flow = float(flow)
                char = None
                if flow <= 0:
                    char = WaterQualityView.FLOWCHARS[0]
                elif flow < 3.6:
                    char = WaterQualityView.FLOWCHARS[1]
                elif flow > 4.7:
                    char = WaterQualityView.FLOWCHARS[2]
                if char is not None:
                    painter.setFont(self.flowfont)
                    painter.drawText(x, y + size, char)
                painter.restore()
            y += size
            if y > maxY:
                break


    def mousePressEvent(self, event):
        fm = QFontMetrics(self.font())
        self.selectedRow = event.y() // fm.height()
        self.update()
        #self.emit(SIGNAL("clicked(QModelIndex)"),
        #          self.model.index(self.selectedRow, 0))
        self.clicked.emit(self.model.index(self.selectedRow, 0))



    def keyPressEvent(self, event):
        if self.model is None:
            return
        row = -1
        if event.key() == Qt.Key_Up:
            row = max(0, self.selectedRow - 1)
        elif event.key() == Qt.Key_Down:
            row = min(self.selectedRow + 1, self.model.rowCount() - 1)
        if row != -1 and row != self.selectedRow:
            self.selectedRow = row
            if self.scrollarea is not None:
                fm = QFontMetrics(self.font())
                y = fm.height() * self.selectedRow
                print(y)
                self.scrollarea.ensureVisible(0, y)
            self.update()
            #self.emit(SIGNAL("clicked(QModelIndex)"),
            #          self.model.index(self.selectedRow, 0))
            self.clicked.emit(self.model.index(self.selectedRow, 0))
        else:
            QWidget.keyPressEvent(self, event)


class MainForm(QDialog):

    def __init__(self, parent=None):
        super(MainForm, self).__init__(parent)

        self.model = WaterQualityModel(os.path.join(
                os.path.dirname(__file__), "waterdata.csv.gz"))
        self.tableView = QTableView()
        self.tableView.setAlternatingRowColors(True)
        self.tableView.setModel(self.model)
        self.waterView = WaterQualityView()
        self.waterView.setModel(self.model)
        scrollArea = QScrollArea()
        scrollArea.setBackgroundRole(QPalette.Light)
        scrollArea.setWidget(self.waterView)
        self.waterView.scrollarea = scrollArea

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tableView)
        splitter.addWidget(scrollArea)
        splitter.setSizes([600, 250])
        layout = QHBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

        self.setWindowTitle("Water Quality Data")
        QTimer.singleShot(0, self.initialLoad)


    def initialLoad(self):
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        splash = QLabel(self)
        pixmap = QPixmap(os.path.join(os.path.dirname(__file__),
                "iss013-e-14802.jpg"))
        #print(os.path.join(os.path.dirname(__file__),
        #        "iss013-e-14802.jpg"))
        splash.setPixmap(pixmap)
        splash.setWindowFlags(Qt.SplashScreen)
        splash.move(self.x() + ((self.width() - pixmap.width()) / 2),
                    self.y() + ((self.height() - pixmap.height()) / 2))
        splash.show()
        QApplication.processEvents()
        try:
            self.model.load()
        except IOError as e:
            QMessageBox.warning(self, "Water Quality - Error", e)
        else:
            self.tableView.resizeColumnsToContents()
        splash.close()
        QApplication.processEvents()
        QApplication.restoreOverrideCursor()


app = QApplication(sys.argv)
form = MainForm()
form.resize(850, 620)
form.show()
app.exec_()
