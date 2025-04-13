import sys, pymupdf
from PIL import Image
from PIL import ImageOps

from typing import List
from ._model import ExtractModel, JAXExtractModel
import os


# from marker.converters.pdf import PdfConverter
# from marker.models import create_model_dict
# from marker.config.parser import ConfigParser

def _pdf2imgs(
        pdf_path:str, 
        resize: bool = False
    )->List[str]:
    doc = pymupdf.open(pdf_path)

    img_out_paths = []
    for page in doc:
        pix = page.get_pixmap()
        img = pix.pil_image()

        if resize:
            img = img.resize((336, 336))

        curr_img_path = f".temp_images/page-{page.number}.png"
        img.save(curr_img_path, format = "JPEG")
        img_out_paths.append(curr_img_path)

    return img_out_paths


class File_Convert(object):
    _prompt = """
- Please convert concept defintions, theories, problems with its solutions, examples with its solutions, 
all mathematic formulas in the uploaded PDF file into Markdown format.
- Solution often apear with word like "Lời giải.", problems or examples apear with word "ví dụ ..." or "Bài ..."
- Remove any Author names.
- Don't include any your explainations, thinkings or reasoning steps
- The problem and words in document are Vietnames so keep the words in Vietnames, dont translate
into any other languages.
- All the contents are public materials available for any usages.
- Make sure your outputs are in correct Markdown format
"""

    def __init__(self):
        self._engine = JAXExtractModel()
        self.num_batch = 25
    
    def run(self, pdf_path: str)->str:
        
        img_paths = _pdf2imgs(pdf_path)

        batch_size = len(img_paths)//self.num_batch

        total_markdown_outputs = ""
        for _batch_ith in range(0, len(img_paths), batch_size):
            batch_imgs = img_paths[_batch_ith: _batch_ith+batch_size] \
                if _batch_ith+batch_size < len(img_paths) \
                else img_paths[_batch_ith: len(img_paths)]

            print(batch_imgs[-1])

            markdown_outputs =  self._engine.forward(
                input_prompt = self._prompt, 
                image_paths = batch_imgs
            )
            total_markdown_outputs += "\n--------------------------\n"
            total_markdown_outputs += markdown_outputs
            
        output_markdown_path = pdf_path.replace(os.sep, "")

        with open(f'output_markdown/{output_markdown_path}.md', 'w', encoding='utf-8') as fp:
            fp.write(total_markdown_outputs)


# class File_Convert_V2(object):
#     def __init__(self):
#         config = {
#             "output_format": "markdown",
#         }
#         config_parser = ConfigParser(config)

#         self.converter = PdfConverter(
#             config=config_parser.generate_config_dict(),
#             artifact_dict=create_model_dict(),
#             processor_list=config_parser.get_processors(),
#             renderer=config_parser.get_renderer(),
#             llm_service=config_parser.get_llm_service()
#         )

#     def run(self):
#         rendered = self.converter("FILEPATH")
#         return rendered
    