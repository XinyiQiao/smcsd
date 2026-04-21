# Tech Design: `generate_particles()` API

## Goal

Add a new public API `SMCEngine.generate_particles()` that returns the outputs of **all N particles** for each request.

---

## Proposed API

```python
engine = SMCEngine(model_path=..., draft_model_path=..., n_particles=8, ...)

# Single prompt → List[ParticleResult] (one entry per particle)
results = engine.generate_particles("What is 2+2?")

# Batch prompt → List[List[ParticleResult]]
results = engine.generate_particles(["prompt A", "prompt B"])
```

### `ParticleResult` schema

```python
{
    "particle_idx":       int,        
    "text":               str,        
    "output_ids":         List[int],  
    "log_weight":         float,      
    "completion_tokens":  int,
    "finished":           bool,      
}
```

The list is ordered by `particle_idx`.

---

## Affected Components

```
smcsd/engine.py                     <- new generate_particles() method + recv loop
smcsd/v2/req_state.py               <- new finalize_group_all_particles()
smcsd/v2/scheduler.py               <- new _finalize_group_all_particles(), flag routing
smcsd/common/io_struct.py  (new)    <- SMCParticlesOutput message type
```

---

## Design

### 1. New ZMQ message type — `smcsd/common/io_struct.py`

`stream_output` serialises `Req` objects into `BatchTokenIDOutput` (one token stream per `rid`). This structure cannot carry per-particle data.

Add a new ZMQ message type to receive all particles from the scheduler subprocess

```python
@dataclasses.dataclass
class SMCParticlesOutput:
    """Scheduler -> engine: all-particle result for one request."""
    rid:        str
    particles:  List[dict]   # list of ParticleResult dicts 
```

### 2. `ScheduleBatchSMC.finalize_group_all_particles()` — `smcsd/v2/req_state.py`

Parallel to the existing `finalize_group()`.

1. Reads `group_log_weights[group_id]` and `group_slot_lists[group_id]`
2. For every slot in the group, reads `slot_to_req[slot].output_ids`, `token_counts[slot]`, `finished_mask[slot]`, `particle_indices[slot]`
3. Builds a `List[dict]` of `ParticleResult`s
4. Calls `free_group_slots(group_id)` (same as `finalize_group`)
5. Returns `(parent_req, List[dict])` — parent_req is needed to carry `rid` and `time_stats`

The existing `finalize_group` is unchanged; this is additive.

### 3. Scheduler routing — `smcsd/v2/scheduler.py`

`SMCSchedulerV2` needs to know whether a group was submitted via `generate()` or `generate_particles()`.

Add one field in `SequenceGroup`

```python
@dataclass
class SequenceGroup:
    parent_req: Req
    n_particles: int
    particle_temperature: float
    return_all_particles: bool = False   # <- new
    ...
```

`generate_particles()` set it on `SamplingParams.json_schema`

```python
# engine.py — generate_particles() sets this before sending
sp.json_schema = "__smc_return_all_particles__"
```

```python
return_all_particles = (
    getattr(req.sampling_params, "json_schema", None) == "__smc_return_all_particles__"
)
group = SequenceGroup(
    parent_req=req,
    n_particles=self.server_args.smc_n_particles,
    particle_temperature=self.server_args.smc_draft_temperature,
    return_all_particles=return_all_particles,
)
```

`_finalize_group` branches on `group.return_all_particles`:

```python
def _finalize_group(self, group: SequenceGroup) -> None:
    if group.return_all_particles:
        self._finalize_group_all_particles(group)
    else:
        parent_req = self.slot_state.finalize_group(group.group_id, group.parent_req)
        parent_req.time_stats.set_completion_time()
        self.stream_output([parent_req], False)

def _finalize_group_all_particles(self, group: SequenceGroup) -> None:
    parent_req, particles = self.slot_state.finalize_group_all_particles(
        group.group_id, group.parent_req
    )
    parent_req.time_stats.set_completion_time()
    msg = SMCParticlesOutput(rid=parent_req.rid, particles=particles)
    self.send_to_tokenizer.send_pyobj(msg)
```

### 4. `SMCEngine.generate_particles()` — `smcsd/engine.py`

- Sets `sp.json_schema = "__smc_return_all_particles__"` on each request's `SamplingParams`
- The recv loop waits for `SMCParticlesOutput` instead of `BatchTokenIDOutput`
- Returns `List[ParticleResult]` (single) or `List[List[ParticleResult]]` (batch)

```python
def generate_particles(self, prompt=None, sampling_params=None, input_ids=None):
    # ... same normalisation as generate() ...

    for text, ids, sp_dict in zip(prompts, ids_list, sampling_params_list):
        sp = SamplingParams(**sp_dict) if isinstance(sp_dict, dict) else sp_dict
        sp.normalize(self.tokenizer)
        sp.json_schema = "__smc_return_all_particles__"   # sentinel

        req = TokenizedGenerateReqInput(
            rid=rid, input_text=text, input_ids=ids, sampling_params=sp, ...
        )
        self.send_to_scheduler.send_pyobj(req)

    pending = set(rids)
    results = {}
    while pending:
        msg = self._recv_scheduler_output()
        if isinstance(msg, SMCParticlesOutput) and msg.rid in pending:
            results[msg.rid] = msg.particles
            pending.discard(msg.rid)
        elif isinstance(msg, AbortReq) and msg.rid in pending:
            results[msg.rid] = []
            pending.discard(msg.rid)

    outputs = [results[rid] for rid in rids]
    return outputs[0] if is_single else outputs
```


