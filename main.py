from flask import Flask, render_template, request

from utils import extract_actions


def load_list_from_file(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().splitlines()


app = Flask(__name__)

CHEMICALS_FILE_PATH = "data/chemicals.txt"
USAGE_ACTIONS_FILE_PATH = "data/usage_actions.txt"
THRESHOLD = 0.5

chemicals = load_list_from_file(CHEMICALS_FILE_PATH)
usage_actions = load_list_from_file(USAGE_ACTIONS_FILE_PATH)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        actions = extract_actions(text, chemicals, usage_actions, THRESHOLD)
        print(actions)
        return render_template('index.html', text=text, actions=actions)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
