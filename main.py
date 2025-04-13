from rag_math import File_Convert
import os, glob

if __name__ == "__main__":
    for f in glob.glob(".temp_images/*.png"):
        os.remove(f)

    engine = File_Convert()

    _re = engine.run("datasets\\theories\\tong-hop-ly-thuyet-thpt-mon-toan-tran-thanh-yen.pdf")

    print(_re)