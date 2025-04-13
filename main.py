from rag_math import File_Convert
import os, glob

if __name__ == "__main__":
    for f in glob.glob(".temp_images/*.png"):
        os.remove(f)

    engine = File_Convert()

    _re = engine.run("datasets\\theories\\ly-thuyet-cac-dang-toan-va-bai-tap-cung-va-goc-luong-giac-cong-thuc-luong-giac.pdf")


    print(_re)