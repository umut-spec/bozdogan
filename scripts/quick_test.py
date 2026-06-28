"""Test - 20 istek, 10 paralel"""
import sys
sys.path.insert(0, '.')

import asyncio
from generate_massive_dataset import MassiveDatasetGenerator

async def main():
    generator = MassiveDatasetGenerator(
        total_requests=20,
        concurrent_limit=10
    )

    await generator.generate_all()

if __name__ == "__main__":
    asyncio.run(main())
