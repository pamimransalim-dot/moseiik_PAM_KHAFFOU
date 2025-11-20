FROM rust:latest
WORKDIR /app
COPY . .
RUN cargo build --release
ENTRYPOINT ["cargo", "test", "--release", "--"]
