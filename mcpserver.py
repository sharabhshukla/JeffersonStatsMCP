import asyncio
from fastmcp import FastMCP
from fastmcp.server.server import Request, Response
from src.basics import basic_stats_mcp



mcp = FastMCP(
    name="JeffersonMCP",
    version="0.1.0",
    host="0.0.0.0",
    port=8080,

)


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request: Request) -> Response:
    return Response(content="OK", status_code=200, headers={"Content-Type": "text/plain"})


async def setup_main_server():
    await mcp.import_server(server=basic_stats_mcp, prefix="/basic-stats")


if __name__ == '__main__':
    asyncio.run(setup_main_server())
    mcp.run(
        transport="http"
    )
