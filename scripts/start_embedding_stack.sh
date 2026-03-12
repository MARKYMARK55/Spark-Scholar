
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "       Restarting (Embedding Stack)                               "
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


docker stop vllm_node


cd ~
cd vllm/model_stack/start_stack

./start_core_services_small.sh


cd ~
cd vllm/model_stack/router_embed_rerank

docker compose -f start_phi_4_mini.yml up -d
docker compose -f start_bge-m3_embedder.yml down

docker compose -f start_bge-m3_embedder_indexing.yml down

docker compose -f start_bge-m3_embedder_indexing.yml up -d 



echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "          Startup Complete                "
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"