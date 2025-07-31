from neo4j import GraphDatabase
from typing import List, Callable

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

    def reset(self, initial_prompt: str) -> str:
        self.history, self.trajectory = [], []
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
        obs = f"Top candidates (use navigate(id)):\n{listing}"

        self.history.append({"role": "user", "content": obs})
        return obs

    def navigate(self, node_id: str):
        self._require_reset()
        details, neigh = self._fetch(node_id)
        if details is None:
            obs, done = f"No node with id '{node_id}'.", False
        else:
            self.curr_node = node_id
            self.trajectory.append(node_id)
            nbs  = ", ".join(n['id'] for n in neigh) or "none"
            text = details.get("content") or details.get("description") or ""
            obs  = (f"Node {details['id']} ({details['type']}):\n"
                    f"label: {details['label']}\n"
                    f"text : {text[:200]}{'...' if len(text)>200 else ''}\n"
                    f"Neighbours: {nbs}")
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
