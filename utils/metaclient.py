import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


class MetadataClient:
    def __init__(self):
        self.client = MongoClient(os.environ.get('MONGODB_URI'))
        self.metadata = self.client['metadata']['datasets']

    def get_scores(self):
        """
        Get scores from the meta storage.
        :return: List of scores.
        """
        return self.metadata.find({})

    def add_dataset(self, props_dict):
        """
        Add metadata to the meta storage.
        :param props_dict: DatasetProps instance.
        :return: None.
        """
        self.metadata.insert_one(props_dict)
