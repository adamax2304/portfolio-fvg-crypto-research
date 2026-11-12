"""
Quick verification that all modules import correctly
and basic functionality works.
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_imports():
    print("Testing imports...")
    from multi_agent import config
    print(f"  config.py OK — FVG composite floor: {config.FVG_AGENT_CONFIG['SIGNAL_MIN_COMPOSITE_SCORE']}")
    print(f"  config.py OK — FVG composite ceiling: {config.FVG_AGENT_CONFIG['SIGNAL_MAX_COMPOSITE_SCORE']}")
    print(f"  config.py OK — SMC SL mode: {config.SMC_AGENT_CONFIG['SL_MODE']}")

    from multi_agent.shared.fvg_core import detect_fvgs_v2, FVG, compute_ema
    print(f"  fvg_core.py OK")

    from multi_agent.shared.state_manager import StateManager
    sm = StateManager()
    await sm.set_agent_status("fvg_agent", "running")
    status = await sm.get_agent_status("fvg_agent")
    print(f"  state_manager.py OK — agent status: {status}")

    from multi_agent.shared.risk_manager import RiskManager
    rm = RiskManager()
    qty, lev = rm.calculate_position_size(1000.0, 100.0, 98.0, "BTC/USDT:USDT", {})
    print(f"  risk_manager.py OK — qty: {qty}, lev: {lev}")

    from multi_agent.filters.oi_delta_filter import OIDeltaFilter
    print(f"  oi_delta_filter.py OK")

    from multi_agent.agents.fvg_agent import FVGAgent
    print(f"  fvg_agent.py OK")

    from multi_agent.agents.smc_agent import SMCAgent
    print(f"  smc_agent.py OK")

    from multi_agent.master_agent import MasterAgent
    print(f"  master_agent.py OK")

    print("\nAll imports OK!")

if __name__ == "__main__":
    asyncio.run(test_imports())
