import sys
from wand.color import Color
from wand.image import Image
from pprint import pprint

import sys
from wand.image import Image

oldfilename = "C:\data\\test\\1.jpg"

with Image(filename=oldfilename) as image:
    himage = image.clone()
    vimage = image.clone()

    # flop() - 좌우뒤집기
    himage.flop()

    # flip() -  상하뒤집기
    vimage.flip()

    himage.save(filename="flop_" + oldfilename)
    vimage.save(filename="flip_" + oldfilename)




pprint(sys.path)