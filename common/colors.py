import attr


@attr.frozen
class AtiumColorsBGR:

    # TODO: Smart way to move between RGB <-> BGR

    BLUE = (255, 128, 0)
    LIGHT_BLUE = (255, 178, 102)
    GREEN = (102, 204, 0)
    LIGHT_GREEN = (153, 255, 51)
    RED = (51, 51, 255)
    LIGHT_RED = (153, 153, 255)

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
