from neo4j import GraphDatabase
from typing import List, Callable, Dict, Any
import re

class Neo4jKGEnv:
    """
    reset(prompt)      -> observation:str
    navigate(node_id)  -> observation:str, done:bool
    stop()             -> observation:str, done:bool
    """
    def __init__(
        self,
        embed_fn: Callable[[str], List[float]],
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_pwd:  str = "password",
        top_k: int = 5,
    ):
        if embed_fn is None:
            raise ValueError("embed_fn must be provided")
        self.embed_fn   = embed_fn
        self.driver     = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pwd))
        self.top_k      = top_k
        self.reset_called = False
        self.end = False
        self.history, self.trajectory, self.cache = [], [], {}

    _NAV = re.compile(r'^navigate\(([A-Za-z0-9_\-\/.#]+)\)$')
    _STOP = re.compile(r"stop\s*\(\s*\)", re.I)

    def parse_cmd(self, text: str) -> Dict[str, Any]:
        text = text.split("</think>")[-1]
        if _STOP.match(text.strip()):
            return {"action": "stop"}
        m = _NAV.match(text.strip())
        if m:
            return {"action": "navigate", "id": m.group(1).strip()}
        return {"action": "invalid"}

    def end(self):
        return self.end

    def move(self, action: str):
        cmd = self.parse_cmd(action)
        if cmd["action"] == "stop":
            self.end = True
            self.stop()
        elif cmd["action"] == "navigate":
            self.navigate(cmd["id"])
        else:
            self.end = True
        return self.history

    def reset(self, initial_prompt: str):
        self.curr_node, self.reset_called = None, True

        vec = self.embed_fn(initial_prompt)

        with self.driver.session() as s:
            docs = s.run(
                "CALL db.index.vector.queryNodes('document_embedding_index',$k,$vec)"
                " YIELD node,score RETURN node{.*,score:score,type:'Document'} AS res",
                k=self.top_k, vec=vec).data()

            secs = s.run(
                "CALL db.index.vector.queryNodes('section_embedding_index',$k,$vec)"
                " YIELD node,score RETURN node{.*,score:score,type:'Section'} AS res",
                k=self.top_k, vec=vec).data()

        def fmt(r):
            r = r["res"]
            return f"{r['id']} | {r['type']} | {r['label']} | score={round(r['score'],3)}"

        listing = "\n".join(fmt(x) for x in docs + secs)

        prompt = f"""
        <introduction>
A user asks a question. Your job is to fetch relevant information to answer the question from a knowledge graph.
</introduction>

<goal>
 You will be given a table having headers (score, label, id), that are plausible candidate nodes for exploration. Start by navigating to these nodes using navigation-options. You can only respond with one navigation action, with no other text.
</goal>

<schema>
This is the schema for the data inside the node:

- type: either of document or category or section.
- content: the content of the section, this property is only present in section node.
- description: the description of the document, holds the purpose of the document.
- label: Name of the document or category or section.
- id: id of the node usually of the pattern - category/document#section#<sub-section: optional>
- links: is a type of array of < id , label >

PS: the nodes are hierarchical, starting from category - document - section - sub-section.

</schema>

<navigation-options>
You have following options available, as response:

- navigate: respond with navigate(node-id) to get the node's data, you should not come up with your one node data, it should be one of the links or should be one that you have previously visited. Node's data is of the schema - (id, label, content or description and links)
- stop: respond with stop() when you are done with collecting the context and answer is satisfactory.
</navigation-options>

<input>
	<question>{initial_prompt}</question>
	<starting-nodes>
        {listing}
	</starting-nodes>
</input>

        """

        self.history.append({"role": "user", "content": prompt})
        return self.history

    def navigate(self, node_id: str):
        self._require_reset()
        details, neigh = self._fetch(node_id)
        if details is None:
            self.trajectory.append("__FAILED__")
            obs, done = f"No node with id '{node_id}'.", True
        else:
            self.curr_node = node_id
            self.trajectory.append(node_id)
            nbs  = ", ".join(n['id'] for n in neigh) or "none"
            text = details.get("content") or details.get("description") or ""
            obs  = (f"Node {details['id']} ({details['type']}):\n"
                    f"label: {details['label']}\n"
                    f"text : {text}\n"
                    f"Neighbours: {nbs}")
            self.cache[node_id] = details
            done = False
        self.history += [{"role":"assistant","content":f"navigate({node_id})"},
                         {"role":"user","content":obs}]
        return obs, done

    def stop(self):
        self._require_reset()
        obs, done = "Episode finished.", True
        self.history += [{"role":"assistant","content":"stop()"},
                         {"role":"user","content":obs}]
        return obs, done

    def _fetch(self, node_id):
        with self.driver.session() as s:
            rec = s.run(
                "MATCH (n {id:$id}) OPTIONAL MATCH (n)--(m)"
                " RETURN n{.*,type:labels(n)[0]} AS n,"
                "        collect(m{.*,type:labels(m)[0],id:m.id}) AS neigh",
                id=node_id).single()
            return (None, []) if rec is None else (rec["n"], rec["neigh"])

    def _require_reset(self):
        if not self.reset_called:
            raise RuntimeError("Call reset() before navigate() or stop().")