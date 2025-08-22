from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError, Neo4jError
from typing import List, Callable, Dict, Any
import time, random, re

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

        self.embed_fn = embed_fn
        self.top_k = top_k
        self.reset_called = False
        self.end = False
        self.history, self.trajectory, self.cache, self.actions = [], [], {}, []

        # save creds for reconnection
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_pwd  = neo4j_pwd

        # create driver with retry + backoff
        self.driver = self._create_driver_with_retry()

    # ---------- retry utilities ----------
    def _sleep_backoff(self, attempt: int, base: float, cap: float) -> None:
        # full jitter: base * 2^attempt, capped, multiplied by random [0,1)
        delay = min(cap, base * (2 ** attempt)) * random.random()
        time.sleep(delay)

    def _create_driver_with_retry(self, max_retries: int = 6, base: float = 0.25, cap: float = 4.0):
        last = None
        for attempt in range(max_retries + 1):
            try:
                drv = GraphDatabase.driver(self._neo4j_uri, auth=(self._neo4j_user, self._neo4j_pwd))
                # Ensure the cluster is actually reachable
                try:
                    drv.verify_connectivity()
                except AttributeError:
                    # Older drivers may not have verify_connectivity; do a quick ping
                    with drv.session() as s:
                        s.run("RETURN 1").consume()
                return drv
            except (ServiceUnavailable, TransientError) as e:
                last = e
                if attempt == max_retries:
                    raise
                self._sleep_backoff(attempt, base, cap)
            except Neo4jError:
                # Non-transient errors (auth, query syntax, etc.) shouldn't be retried.
                raise
        # Should never get here
        raise last if last else RuntimeError("Failed to create Neo4j driver")

    def _run_with_retry(self, func, max_retries: int = 6, base: float = 0.1, cap: float = 2.0):
        last = None
        for attempt in range(max_retries + 1):
            try:
                return func()
            except (ServiceUnavailable, TransientError) as e:
                last = e
                # Recreate driver (connection may be broken)
                try:
                    self.driver.close()
                except Exception:
                    pass
                self.driver = self._create_driver_with_retry()
                if attempt == max_retries:
                    raise
                self._sleep_backoff(attempt, base, cap)
            except Neo4jError:
                # Non-transient: surface immediately
                raise
        raise last if last else RuntimeError("Neo4j operation failed")

    def _query_data(self, cypher: str, **params):
        def work():
            with self.driver.session() as s:
                return s.run(cypher, **params).data()
        return self._run_with_retry(work)

    def _query_single(self, cypher: str, **params):
        def work():
            with self.driver.session() as s:
                return s.run(cypher, **params).single()
        return self._run_with_retry(work)

    # ---------- rest of your env ----------
    def parse_cmd(self, text: str) -> Dict[str, Any]:
        _NAV = re.compile(r'^navigate\(([A-Za-z0-9_\-\/.#]+)\)$')
        _STOP = re.compile(r"stop\s*\(\s*\)", re.I)

        text = text.split("</think>")[-1]
        if _STOP.match(text.strip()):
            self.actions.append(text.strip())
            return {"action": "stop"}
        m = _NAV.match(text.strip())
        if m:
            self.actions.append(text.strip())
            return {"action": "navigate", "id": m.group(1).strip()}
        self.actions.append(text.strip())
        return {"action": "invalid"}

    def ended(self):
        return self.end

    def move(self, action: str):
        cmd = self.parse_cmd(action)
        if cmd["action"] == "stop":
            self.end = True
            self.stop()
            self.history += [{"role":"assistant","content":action}]
        elif cmd["action"] == "navigate":
            obs, done = self.navigate(cmd["id"])
            if done:
                self.end = True
                self.history += [{"role":"assistant","content":action}]
            else:
                self.history += [{"role":"assistant","content":action},
                         {"role":"user","content":obs}]
        else:
            self.end = True
            self.history += [{"role":"assistant","content":action}]
        return self.history

    def reset(self, initial_prompt: str):
        self.curr_node, self.reset_called = None, True
        self.question = initial_prompt

        vec = self.embed_fn(initial_prompt)

        docs = self._query_data(
            "CALL db.index.vector.queryNodes('document_embedding_index',$k,$vec) "
            "YIELD node,score RETURN node{.*,score:score,type:'Document'} AS res",
            k=self.top_k, vec=vec
        )
        secs = self._query_data(
            "CALL db.index.vector.queryNodes('section_embedding_index',$k,$vec) "
            "YIELD node,score RETURN node{.*,score:score,type:'Section'} AS res",
            k=self.top_k, vec=vec
        )

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
        return obs, done

    def stop(self):
        self._require_reset()
        obs, done = "Episode finished.", True
        self.history += [{"role":"assistant","content":"stop()"},
                         {"role":"user","content":obs}]
        return obs, done

    def _fetch(self, node_id):
        rec = self._query_single(
            "MATCH (n {id:$id}) OPTIONAL MATCH (n)--(m) "
            "RETURN n{.*,type:labels(n)[0]} AS n, "
            "       collect(m{.*,type:labels(m)[0],id:m.id}) AS neigh",
            id=node_id
        )
        return (None, []) if rec is None else (rec["n"], rec["neigh"])

    def _require_reset(self):
        if not self.reset_called:
            raise RuntimeError("Call reset() before navigate() or stop().")

    def __del__(self):
        try:
            self.driver.close()
        except Exception:
            pass
