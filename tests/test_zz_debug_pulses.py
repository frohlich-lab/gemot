import asyncio, time
from gemot.agent import driver as D
import test_stall_pulses as T

def test_debug():
    async def run():
        await T._model_call([])
        pulses = D._llm_pulses()
        n = []
        orig = pulses.on_llm_end
        async def counted(*a, **k):
            n.append(time.monotonic()); return await orig(*a, **k)
        pulses.on_llm_end = counted
        t0 = time.monotonic()
        try:
            r = await D._next_step(T._step(T._STEP_S, callbacks=[pulses]).__aiter__(), T._WINDOW_S, pulses=pulses)
            print("ok", r, len(n))
        except D._StepStalled:
            print("stalled after", round(time.monotonic()-t0, 2), "calls", len(n), [round(x-t0,2) for x in n][:5])
    asyncio.run(run())
