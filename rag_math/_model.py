
from google import genai
from google.genai import types

from PIL import Image
from typing import List
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod

from gemma import gm
import numpy as np

class _BaseModel(ABC):
    def __init__(self):
        super().__init__()
        # load_dotenv()
        # self._client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

    @abstractmethod
    def forward(self):
        ...

class ExtractModel(_BaseModel):
    def __init__(self):
        super().__init__()

    def forward(self, input_prompt: str, image_paths: List[str] = None, pdf_file:str = None)->str:
        # image_paths = image_paths[3:]
        file_upload = self._client.files.upload(file=pdf_file)
        total_parts = [file_upload]
        # total_parts = [Image.open(_img_path)
        #              for _img_path in image_paths
        # ]
        
        total_parts.append(input_prompt)

        response = self._client.models.generate_content(
            # model = 'gemini-1.5-flash',
            model = 'gemini-2.5-pro-exp-03-25',
            config = types.GenerateContentConfig(
                system_instruction = "You are a professtional mathematics in document understanding.",
                max_output_tokens = 30000,
                temperature = 0.1,
                # safety_settings=[
                #     types.SafetySetting(
                #         category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                #         threshold=types.HarmBlockThreshold.BLOCK_NONE,
                #     ),
                # ],
            ),
            contents = total_parts,
        )
        # print('response: ', response, "\n",response.text)
        assert response.text is not None, f"{response}"

        self._client.files.delete(name=file_upload.name)
        return response.text


class JAXExtractModel(_BaseModel):
    r"""
    Using JAX for runing gemma3-4b-it
    """
    def __init__(self):
        super().__init__()
        
        # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.95"

        model = gm.nn.Gemma3_4B()
        params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

        self._sampler = gm.text.ChatSampler(
            model = model,
            params = params,
            max_out_length = 2048
        )

    def forward(self, input_prompt: str, image_paths: List[str]):
        # image pre-processing
        images_list = [np.array(Image.open(_img_path))
                  for _img_path in image_paths
                  ]

        pad_img_tokens = '\n'.join(['<start_of_image>']*len(image_paths))

        out = self._sampler.chat(
            f'{input_prompt}: \n{pad_img_tokens}',
            images = images_list,
        )
        return out