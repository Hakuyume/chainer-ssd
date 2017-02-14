class Rect(tuple):
    __slots__ = list()

    def __new__(cls, rect):
        return super().__new__(cls, rect)

    def LTRB(left, top, right, bottom):
        return Rect((left, top, right, bottom))

    def LTWH(left, top, width, height):
        return Rect((
            left,
            top,
            left + width,
            top + height))

    def XYWH(center_x, center_y, width, height):
        return Rect((
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2))

    @property
    def left(self):
        return self[0]

    @property
    def top(self):
        return self[1]

    @property
    def right(self):
        return self[2]

    @property
    def bottom(self):
        return self[3]

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    def __and__(self, other):
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)

        if left < right and top < bottom:
            return Rect.LTRB(left, top, right, bottom)
        else:
            return None

    def scale(self, kx, ky=None):
        if ky is None:
            ky = kx
        return Rect.LTRB(
            self.left * kx,
            self.top * ky,
            self.right * kx,
            self.bottom * ky)

    @property
    def area(self):
        return self.width * self.height

    def iou(self, other):
        intersect = self & other
        if intersect is None:
            return 0
        return intersect.area \
            / (self.area + other.area - intersect.area)
