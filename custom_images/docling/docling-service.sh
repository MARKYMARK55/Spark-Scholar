#!/bin/bash

SERVICE_NAME="docling"
IMAGE="spark-docling:debug"
HOST_PORT=8040
CONTAINER_PORT=5001

case "$1" in
    start)
        echo "🚀 Starting Docling service..."
        docker run --gpus all -d \
            --name $SERVICE_NAME \
            -p $HOST_PORT:$CONTAINER_PORT \
            --restart unless-stopped \
            -e CUDA_VISIBLE_DEVICES=0 \
            -e NVIDIA_VISIBLE_DEVICES=all \
            -e DOCLING_DEVICE=cuda \
            -e OMP_NUM_THREADS=8 \
            -e MKL_NUM_THREADS=8 \
            $IMAGE
        echo "✅ Docling service started on port $HOST_PORT"
        ;;
    
    stop)
        echo "🛑 Stopping Docling service..."
        docker stop $SERVICE_NAME
        docker rm $SERVICE_NAME
        echo "✅ Docling service stopped"
        ;;
    
    restart)
        echo "🔄 Restarting Docling service..."
        $0 stop
        sleep 2
        $0 start
        ;;
    
    status)
        if docker ps | grep -q $SERVICE_NAME; then
            echo "✅ Docling service is RUNNING"
            echo "📊 Container info:"
            docker ps --filter "name=$SERVICE_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            echo -e "\n🎮 GPU Status:"
            docker exec $SERVICE_NAME python3 -c "
import torch
print(f'   CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
"
        else
            echo "❌ Docling service is STOPPED"
        fi
        ;;
    
    logs)
        docker logs -f $SERVICE_NAME
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
