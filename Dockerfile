# Image de base Rust
FROM rust:latest

WORKDIR /app

# Copier tout le projet
COPY . .


ENTRYPOINT ["cargo", "test", "--release", "--"]
