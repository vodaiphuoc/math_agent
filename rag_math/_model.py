
from google import genai
from google.genai import types

from PIL import Image
from typing import List
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod

class _BaseModel(ABC):
    def __init__(self):
        super().__init__()
        load_dotenv()
        self._client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

    @abstractmethod
    def forward(self):
        ...

class ExtractModel(_BaseModel):
    def __init__(self):
        super().__init__()

    def forward(self, input_prompt: str, image_paths: List[str])->str:
        # image_paths = image_paths[3:]
        
        total_parts = [Image.open(_img_path)
                     for _img_path in image_paths
        ]
        
        total_parts.append(input_prompt)

        response = self._client.models.generate_content(
            model = 'gemini-1.5-flash',
            config = types.GenerateContentConfig(
                system_instruction = "You are a professtional mathemtics in document understanding.",
                max_output_tokens = 8192,
                temperature = 0.2,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
            ),
            contents = total_parts,
        )
        # print('response: ', response, "\n",response.text)
        assert response.text is not None
        return response.text
    