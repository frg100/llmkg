import os
import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from abc import ABC, abstractmethod

class KnowledgeGraph(ABC):
    def __init__(self, name=None, verbose = 0):
        self._name = name
        self._verbose = verbose
        
        self._entities = [] # list(string)
        self._relations = [] # list(string)
        # np.array([(head_entity, relation, tail_entity)])
        self._triples = np.zeros(shape=(0,3))
        
        self._built = False
        
    ####### PUBLIC #######
    @property
    def name(self):
        return self._name
    
    @property
    def entities(self):
        return self._entities
    
    @property
    def relations(self):
        return self._relations
    
    @property
    def triples(self):
        return self._triples
    
    def sample(self, k=1, negative=False):
        if negative:
            return self._sample_negative_loose(k)
        else:
            return self._sample_positive(k)
        
    def sample_english(self, k=1, negative=False):
        samples = self.sample(k, negative)
        
        english_samples = []
        for sample in samples:
            head_idx, relation_idx, tail_idx = sample
            head_id, relation_id, tail_id = self._entities[head_idx], self._relations[relation_idx], self._entities[tail_idx]
            head, relation, tail = self._id2entity(head_id), self._id2relation(relation_id), self._id2entity(tail_id)
            english_samples.append(relation.replace("{HEAD}", head).replace("{TAIL}", tail))
            
        return english_samples
            
        
    ####### PRIVATE #######
    
    @abstractmethod
    def _id2entity(self, eid):
        """
        A function that maps an entity id (eid) stored in the
        self._entities structure to an english identifier
        and/or description.
        """
        
    @abstractmethod
    def _id2relation(self, rid):
        """
        A function that maps an relation id (rid) stored in
        the self._relations structure to an english identifier
        and/or description.
        """
    
    @abstractmethod
    def _build_graph(self):
        """
        A function that builds the graph by reading in the data in
        its current format and populating self._entities, self._relations,
        self._triples, and at the end should set self._built to True.
        """
        pass
    
    @property
    def _is_built(self):
        return self._built
    
    @property
    def _num_entities(self):
        return len(self._entities)
    
    @property
    def _num_relations(self):
        return len(self._relations)
    
    @property
    def _num_triples(self):
        return self._triples.shape[0]
    
    def _validate_graph(self):
        # Make sure properties are filled out
        assert self._built, "The graph is not built. Please build " \
        "or check that your build_graph method sets self._build " \
        "to True after completion"
        
        # Make sure shape of self._triples is [N, 3]
        assert self._triples.shape[1] == 3, "The _triples property" \
        "must have a shape of 3 in the second dimension. " \
        
        # Make sure all head, tail entities and relations are valid
        head_entities = self._triples[:,0]
        assert head_entities.max() <= len(self._entities), "There" \
        "exists an entity in the head entities of the _triples " \
        "property that exceeds the number of available entities." \
        
        tail_entities = self._triples[:,2]
        assert tail_entities.max() <= len(self._entities), "There " \
        "exists an entity in the tail entities of the _triples " \
        "property that exceeds the number of available entities." \
        
        relations = self._triples[:,1]
        assert relations.max() <= len(self._relations), "There " \
        "exists an relations in the _triples " \
        "property that exceeds the number of available relations." \
        
        for eid in self._entities:
            assert self._id2entity(eid), f"One of the entities ({eid}) " \
            "has no mapping."
            
        for rid in self._relations:
            assert self._id2relation(rid), f"One of the relations ({rid}) " \
            "has no mapping."
            
        assert self.sample(10).shape == (10, 3), "Sampling yields the " \
            "wrong shape"
        
        assert self.sample(10, negative=True).shape == (10, 3), "Sampling " \
            "yields the wrong shape"
        
        if self._verbose >= 1:
            print("Graph was successfully validated!")
        
    def _sample_positive(self, k):
        triple_indices = np.random.choice(self._num_triples, k)
        positive_samples = self._triples[triple_indices]
        
        return positive_samples
    
    def _sample_negative_loose(self, k):
        # TODO(frg100): Make a strict version that makes sure not to
        # add existing triples
        head_entities = np.expand_dims(np.random.choice(self._num_entities, k), 0)
        relations = np.expand_dims(np.random.choice(self._num_relations, k), 0)
        tail_entities = np.expand_dims(np.random.choice(self._num_entities, k), 0)
        
        negative_samples = np.concatenate([head_entities, relations, tail_entities], axis=0).T
        
        return negative_samples
    
    def _load_json_mapping(self, json_path):
        # Load the map
        with open(json_path) as json_file:
            return json.load(json_file)
    
class FB15k237(KnowledgeGraph):
    def __init__(self, base_path=None, splits=['train', 'test', 'valid'], verbose = 0):
        super().__init__(name='FB15k-237', verbose = verbose)
        
        self._base_path = base_path
        self._splits = splits
        
        self._entity_mapping = None
        self._relation_mapping = None
        
        start = time.time()
        self._build_graph(verbose)
        end = time.time()
        if verbose >= 1:
            print(f"Building the graph took {round(end-start)} seconds")    
        
            
    def _id2entity(self, eid):
        if self._entity_mapping is None:
            assert False, "Entity mapping must be populated"
            
        if eid not in self._entity_mapping:
            #print(f"Entity with id ({eid}) is not mapped...")
            return None
            
        return self._entity_mapping[eid]['label']
    
    def _id2relation(self, rid):
        if self._relation_mapping is None:
            assert False, "Relation mapping must be populated"
            
        if rid not in self._relation_mapping:
            #print(f"Relation with id ({rid}) is not mapped...")
            return None
            
        return self._relation_mapping[rid]

    def _build_graph(self, verbose):
        # Load the mappings
        id2entity_path = os.path.join(self._base_path, "entity2wikidata.json")
        self._entity_mapping = self._load_json_mapping(id2entity_path)
        id2relation_path = os.path.join(self._base_path, "relation_mapping.json")
        self._relation_mapping = self._load_json_mapping(id2relation_path)
        
        # Initialize data structures for bookkeeping
        entities = set()
        relations = set()
        triples = set()

        num_data_points = sum(sum(1 for line in open(os.path.join(self._base_path, f"{split}.txt"))) for split in self._splits)
        
        # Load data
        for split in self._splits:
            path = os.path.join(self._base_path, f"{split}.txt")
            if verbose >= 1:
                print(f"Loading file {split}.txt")
                
            # Process into entities, relations, and triples
            with open(path, 'r') as f:
                for line in f:
                    # Check progress
                    last_percent_done = round((100*(self._num_triples-1))/num_data_points)
                    percent_done = round((100*self._num_triples)/num_data_points)
                    if verbose >= 2 and percent_done % 5 == 0 and last_percent_done % 5 != 0:
                        print(f"Data loading progress: [{percent_done}%]")
                    
                    # Initialize data
                    head, relation, tail = line.split()
                    head_id, relation_id, tail_id = None, None, None
                    
                    # If either of the entities has no natural language translation,
                    if not self._id2entity(head) or not self._id2entity(tail):
                        # Don't process it
                        continue
                    
                    if verbose >= 3 and percent_done % 5 == 0 and last_percent_done % 5 != 0:
                        print(f"{self._id2entity(head)} {relation} {self._id2entity(tail)}")
                    
                    # Process head
                    if head not in entities:
                        entities.add(head)
                        head_id = len(self._entities)
                        self._entities.append(head)
                    else:
                        head_id = self._entities.index(head)
                     
                    # Process tail
                    if tail not in entities:
                        entities.add(tail)
                        tail_id = len(self._entities)
                        self._entities.append(tail)
                    else:
                        tail_id = self._entities.index(tail)
                        
                    # Process relation
                    if relation not in relations:
                        relations.add(relation)
                        relation_id = len(self._relations)
                        self._relations.append(relation)
                    else:
                        relation_id = self._relations.index(relation)

                    # Create and add triple
                    triple = np.array([[head_id, relation_id, tail_id]], dtype=np.int32)  
                    if self._num_triples == 0:
                        self._triples = triple
                    else:
                        self._triples = np.append(self._triples, triple, axis=0)
                        
        # Build and validate
        self._built = True
        self._validate_graph()
        

class WN18RR(KnowledgeGraph):
    def __init__(self, base_path=None, splits=['train', 'test', 'valid'], verbose = 0):
        super().__init__(name='WN18RR', verbose = verbose)
        
        self._base_path = base_path
        self._splits = splits
        
        self._relation_mapping = None
        
        start = time.time()
        self._build_graph(verbose)
        end = time.time()
        if verbose >= 1:
            print(f"Building the graph took {round(end-start)} seconds")    
        
            
    def _id2entity(self, eid):
        return eid.split('.')[0].replace('_', ' ')
    
    def _id2relation(self, rid):
        if self._relation_mapping is None:
            assert False, "Relation mapping must be populated"
            
        if rid not in self._relation_mapping:
            #print(f"Relation with id ({rid}) is not mapped...")
            return None
            
        return self._relation_mapping[rid]

    def _build_graph(self, verbose):
        # Load the mappings
        id2relation_path = os.path.join(self._base_path, "relation_mapping.json")
        self._relation_mapping = self._load_json_mapping(id2relation_path)
        
        # Initialize data structures for bookkeeping
        entities = set()
        relations = set()
        triples = set()

        num_data_points = sum(sum(1 for line in open(os.path.join(self._base_path, f"{split}.txt"))) for split in self._splits)
        
        # Load data
        for split in self._splits:
            path = os.path.join(self._base_path, f"{split}.txt")
            if verbose >= 1:
                print(f"Loading file {split}.txt")
                
            # Process into entities, relations, and triples
            with open(path, 'r') as f:
                for line in f:
                    # Check progress
                    last_percent_done = round((100*(self._num_triples-1))/num_data_points)
                    percent_done = round((100*self._num_triples)/num_data_points)
                    if verbose >= 2 and percent_done % 5 == 0 and last_percent_done % 5 != 0:
                        print(f"Data loading progress: [{percent_done}%]")
                    
                    # Initialize data
                    head, relation, tail = line.split()
                    head_id, relation_id, tail_id = None, None, None
                    
                    if verbose >= 3 and percent_done % 5 == 0 and last_percent_done % 5 != 0:
                        print(f"{self._id2entity(head)} {relation} {self._id2entity(tail)}")
                    
                    # Process head
                    if head not in entities:
                        entities.add(head)
                        head_id = len(self._entities)
                        self._entities.append(head)
                    else:
                        head_id = self._entities.index(head)
                     
                    # Process tail
                    if tail not in entities:
                        entities.add(tail)
                        tail_id = len(self._entities)
                        self._entities.append(tail)
                    else:
                        tail_id = self._entities.index(tail)
                        
                    # Process relation
                    if relation not in relations:
                        relations.add(relation)
                        relation_id = len(self._relations)
                        self._relations.append(relation)
                    else:
                        relation_id = self._relations.index(relation)

                    # Create and add triple
                    triple = np.array([[head_id, relation_id, tail_id]], dtype=np.int32)  
                    if self._num_triples == 0:
                        self._triples = triple
                    else:
                        self._triples = np.append(self._triples, triple, axis=0)
                        
        # Build and validate
        self._built = True
        self._validate_graph()
    