import re
import json
import chromadb
import networkx as nx
import ollama

OLLAMA_MODEL= "qwen2.5-coder:3b"

class HyrbidGraphVectorRAG:
    def __init__(self, collection_name="hybrid_rag"):
        self.chroma_client= chromadb.Client()
        self.collection= self.chroma_client.get_or_create_collection(name=collection_name)
        self.graph=nx.DiGraph()

    def add_document(self, doc_text):
        chunks=[c.strip() for c in doc_text.split("\n\n") if c.strip()]

        print(f"Processing {len(chunks)} document chunks")

        for idx, chunk in enumerate(chunks):
            chunk_id= f"doc_chunk_{idx}"

            self.collection.add(
                documents=[chunk],
                ids=[chunk_id]
            )

            print(f"Extracting graph Triplets for Chunk {idx + 1}/{len(chunks)}")
            self._extract_triplets_to_graph(chunk)

    def _extract_triplets_to_graph(self, text):
        prompt=f"""
        Extract key technical entity relationships from this text.
        Output ONLY a JSON array of triplets in this format:
        [
          {{"source": "EntityA", "relation": "DEPENDS_ON", "target": "EntityB"}},
          {{"source": "EntityB", "relation": "USES", "target": "EntityC"}}
        ]

        Text:
        {text}
        """

        try:
            response=ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}]
            )
            content= response['message']['content'].strip()

            if "```json" in content:
                content= content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content= content.split("```")[1].strip()

            triplets= json.loads(content)
            for t in triplets:
                src, rel, tgt= t.get("source"), t.get("relation"), t.get("target")
                if src and tgt and rel:
                    self.graph.add_node(src, type="Entity")
                    self.graph.add_node(tgt, type="Entity")
                    self.graph.add_edge(src, tgt, relation=rel)

        except Exception as e:
            print(f"Triplet extraction skipped for chunk due to parsing: {e}")

    def vector_search(self, query, top_k=2):
        results=self.collection.query(query_texts=[query], n_results=top_k)
        return results['documents'][0] if results['documents'] else []

    def graph_search(self, query):
        matched_content=[]
        for node in self.graph.nodes():
            if str(node).lower() in query.lower():
                neighbors= list(self.graph.successors(node))
                for n in neighbors:
                    edge_data= self.graph.get_edge_data(node, n)
                    rel=edge_data.get('relation', 'RELATED_TO') if edge_data else 'RELATED_TO'
                    matched_content.append(f"({node}) --[{rel}]--> [{n}]")
        return matched_content

    def hybird_query(self, query):
        print(f"\n Executing Hybrid Retrieval for: '{query}")

        vector_docs= self.vector_search(query, top_k=2)
        graph_facts= self.graph_search(query)

        print("\n--- 📍 Vector Context Retrieved ---")
        for v in vector_docs:
            print(f"• {v}")

        print("\n--- 🕸️ Knowledge Graph Context Traversed ---")
        for g in graph_facts:
            print(f"• {g}")

        synthesis_prompt = f"""
        Answer the following question using BOTH the vector search chunks and graph facts provided.

        Question: {query}

        Vector Search Context:
        {chr(10).join(vector_docs)}

        Knowledge Graph Facts:
        {chr(10).join(graph_facts)}

        Provide a concise, highly accurate synthesis answer:
        """

        response= ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': synthesis_prompt}]
        )

        return response['message']['content']