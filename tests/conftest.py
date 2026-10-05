import os
import sys

# Ļauj testos importēt `src` pakotni, palaižot pytest no repozitorija saknes.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
