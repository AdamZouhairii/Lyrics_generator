import json

class DataFormatter:
    """
    A class that formats data for a specific purpose.

    Args:
        input_file (str): The path to the input file.
        output_file (str): The path to the output file.

    Attributes:
        input_file (str): The path to the input file.
        output_file (str): The path to the output file.
    """

    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def load_data(self):
        """
        Loads data from the input file.

        Returns:
            dict: The loaded data as a dictionary.
        """
        with open(self.input_file, 'r') as f:
            return json.load(f)

    def format_data(self, json_data):
        """
        Formats the given JSON data.

        Args:
            json_data (dict): The JSON data to be formatted.

        Returns:
            list: The formatted data as a list of dictionaries.
        """
        formatted_data = []

        for item in json_data:
            if "artist" in item and item["artist"] == None:
                item["artist"] = "Unknown"

            user_request = {
                "role": "user",
                "content": f"Generate {item['category']} type lyrics  like {item['artist']}"
            }

            assistant_response = {
                "role": "assistant",
                "content": item["lyrics"]
            }

            formatted_data.append(user_request)
            formatted_data.append(assistant_response)

        return formatted_data

    def save_data(self, formatted_data):
        """
        Saves the formatted data to the output file.

        Args:
            formatted_data (list): The formatted data to be saved.
        """
        with open(self.output_file, 'w') as f:
            json.dump(formatted_data, f, indent=2)

        print("Données reformattées écrites dans", self.output_file)

# Usage
formatter = DataFormatter('songs_data.json', 'train.json')
json_data = formatter.load_data()
formatted_data = formatter.format_data(json_data)
formatter.save_data(formatted_data)
