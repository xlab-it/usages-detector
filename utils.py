import difflib
import pickle
from dataclasses import dataclass
from typing import Tuple, List

import spacy


@dataclass
class Context:
    chemicals: list[str]
    usage_actions: list[str]
    threshold: float = 0.7


class ChemicalFinder:
    def __init__(self, context: Context):
        self.context = context
        self.nlp = spacy.load("ru_core_news_sm")

    def find_chemicals(self, text):
        doc = self.nlp(text)
        found = []
        for token in doc:
            if token.text.lower() in self.context.chemicals:
                found.append(token)
            else:
                label, ratio = self.find_most_similar_token(token)
                if ratio > self.context.threshold:
                    found.append(token)
        return found

    def find_most_similar_token(self, token):
        similarities = [(label, difflib.SequenceMatcher(None, token.text.lower(), label).ratio()) for label in
                        self.context.chemicals]
        max_similarity = max(similarities, key=lambda x: x[1])
        return max_similarity

    def find_most_similar(self, target: str, labels_index: list[str]):
        similarities = [(label, difflib.SequenceMatcher(None, target.lower(), label).ratio()) for label in
                        labels_index]
        max_similarity = max(similarities, key=lambda x: x[1], default=(None, 0))

        if max_similarity[1] < self.context.threshold:
            return None, None
        return max_similarity

    def normalize_chemical_label(self, chemical_label: str):
        label = chemical_label.lower()
        label, ratio = self.find_most_similar(label, self.context.chemicals)
        return label

    def normalize_action_label(self, action_label: str):
        label = action_label.lower()
        label, ratio = self.find_most_similar(label, self.context.usage_actions)
        return label


def tokenize_sentences(text):
    with open('tokenizers/punctuation_marks.pickle', 'rb') as f:
        model = pickle.load(f)
        sentences = model.tokenize(text)
        return sentences


def find_head(chemical_token):
    head = chemical_token
    while head != head.head:
        head = head.head
    return head


# в комментариях ниже слово "реактив" - это chemical_token.
def find_quantities(chemical_token):
    def find_head_nmods():
        head = chemical_token
        while head != head.head:  # 15 < [child,nummod]- мл <-[head, nmod]- реактива
            head = head.head
            if head.dep_ != "nmod":
                continue
            for child in head.children:
                if child.dep_ == "nummod":
                    return [(head.text, child.text)]  # UNIT, NUMBER

    def find_child_nmods(token):  # добавим реактив (15 мл)
        for child in token.children:
            if child.dep_ != "parataxis":
                continue
            all_quantities = list(child.conjuncts)
            all_quantities.append(child)
            result = []
            for quantities in all_quantities:
                for grandchild in quantities.children:
                    if grandchild.dep_ == "nummod":
                        result.append((quantities.text, grandchild.text))
            return result

    res1 = find_head_nmods()
    res2 = find_child_nmods(chemical_token)
    res = []
    if res1:
        res += res1
    if res2:
        res += res2
    return res


@dataclass
class Dependency:
    action: (Tuple)[str, str]
    obj: spacy.tokens.Token
    meta: list


def parameters_to_json(parameters: List[Tuple[str, str]]) -> dict:
    converted_params = []
    for param in parameters:
        converted_params.append({
            "unit": param[0],
            "quantity": param[1]
        })
    return converted_params


def dependency_to_dict(normalized_action_label, normalized_chemical_label, meta) -> dict:
    if len(meta) != 0:
        meta = parameters_to_json(meta)
        return {
            "action": normalized_action_label,
            "obj": normalized_chemical_label,
            "meta": meta
        }
    else:
        return {
            "action": normalized_action_label,
            "obj": normalized_chemical_label
        }


def extract_actions(text: str, chemicals: list[str], usage_actions: list[str], threshold: float) -> list[dict]:
    context = Context(
        chemicals=[label.lower() for label in chemicals],
        usage_actions=[label.lower() for label in usage_actions],
        threshold=threshold
    )

    sentences = tokenize_sentences(text)
    finder = ChemicalFinder(context)

    res = []
    for sentence in sentences:
        dependencies = []
        find_chemicals = finder.find_chemicals(sentence)

        for chemical in find_chemicals:
            head = find_head(chemical)
            if head.pos_ in ["NOUN", "VERB"]:
                quantities = find_quantities(chemical)
                dependencies.append(Dependency(
                    action=(head.lemma_, head.pos_),
                    obj=chemical,
                    meta=quantities
                ))

        for dep in dependencies:
            normalized_chemical_label = finder.normalize_chemical_label(dep.obj.lemma_)
            normalized_action_label = finder.normalize_action_label(dep.action[0])
            if (normalized_action_label is not None) and (normalized_chemical_label is not None):
                to_dict = dependency_to_dict(normalized_action_label, normalized_chemical_label, dep.meta)
                res.append(to_dict)
    return res
