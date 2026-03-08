@echo off
echo === Semantic Search Demo ===
curl -X DELETE http://localhost:8000/cache
echo.
echo 1/3: First query ^(CACHE MISS^)
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d "{\"query\": \"gun control debate\"}"
echo.
echo 2/3: Similar query ^(CACHE HIT 88%%^)
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d "{\"query\": \"guns control discussion\"}"
echo.
echo 3/3: Cache stats
curl http://localhost:8000/cache/stats
pause
