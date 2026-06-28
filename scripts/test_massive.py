"""Test script - 50 istek"""
import sys
sys.path.insert(0, 'scripts')

import asyncio
from generate_massive_dataset import MassiveDatasetGenerator

async def main():
    # Test: 50 istek, 20 paralel
    generator = MassiveDatasetGenerator(
        total_requests=50,
        concurrent_limit=20
    )

    await generator.generate_all()

if __name__ == "__main__":
    asyncio.run(main())
