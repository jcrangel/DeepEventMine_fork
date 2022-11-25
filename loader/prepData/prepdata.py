"""Load data from brat format and process for entity"""

from collections import OrderedDict
from html import entities

from DeepEventMine.loader.prepData.brat import brat_loader
from DeepEventMine.loader.prepData.sentence import prep_sentence_offsets, process_input
from DeepEventMine.loader.prepData.entity import process_etypes, process_tags, process_entities
import json

def prep_input_data(files_fold, params,sentences=None,json_file=None):
    # load data from *.ann files
    if sentences is not None:
        sentences0 = sentences
        entities0, _ = gen_sent_entities_json(sentences=sentences)

    elif json_file is not None and sentences is None:
        entities0, sentences0 = gen_sent_entities_json(json_file)
    else:
        entities0, sentences0 = brat_loader(files_fold, params)


    # sentence offsets
    sentences1 = prep_sentence_offsets(sentences0)

    # entity
    entities1 = process_etypes(entities0)  # all entity types
    terms0 = process_tags(entities1)  # terms, offset, tags, etypes
    input0 = process_entities(entities1, sentences1, params, files_fold) ##!!

    # prepare for training batch data for each sentence
    input1 = process_input(input0)

    for doc_name, doc in sorted(input0.items(), key=lambda x: x[0]):
        entities = set()
        num_entities_per_doc = 0
        for sentence in doc:
            eids = sentence["eids"]
            entities |= set(eids)
            num_entities_per_doc += len(eids)

        full_entities = set(entities1["pmids"][doc_name]["ids"])
        diff = full_entities.difference(entities)
        if diff:
            print(doc_name, sorted(diff, key=lambda _id: int(_id.replace("T", ""))))

    # entity indices
    g_entity_ids_ = dict()
    for fid, fdata in entities0.items():
        # get max entity id
        eid_ = [eid for eid in fdata['ids'] if not eid.startswith('TR')]
        ids_ = [int(eid.replace('T', '')) for eid in eid_]
        if len(ids_) > 0:
            max_id = max(ids_)
        else:
            max_id = 0
        eid_.append(max_id)
        g_entity_ids_[fid] = eid_

    return {'entities': entities1, 'terms': terms0, 'sentences': sentences1, 'input': input1,
            'g_entity_ids_': g_entity_ids_}


def gen_sent_entities_json(jsonf=None,sentences=None):

    if jsonf is not None:
        with open(jsonf) as json_file:
            sentences = json.load(json_file)


    # sentences = dict(data)

    entities = {pmid: dict([('data', dict([])), ('types', []), ('counted_types', {}), ('ids', []), ('terms', [])])
                for pmid,valus in sentences.items()
    }
    # entities = {}
    return entities,sentences
    
