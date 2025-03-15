FROM postgres:15-alpine

# Install build dependencies
RUN apk add --no-cache \
    git \
    build-base \
    clang \
    llvm \
    postgresql-dev

# Clone and build pgvector
RUN git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make OPTFLAGS="" \
    && make install

# Cleanup
RUN apk del git build-base clang llvm postgresql-dev \
    && rm -rf /pgvector 

# Copy initialization scripts
COPY init.sql /docker-entrypoint-initdb.d/

# Copy data dump
COPY dump.sql /tmp/dump.sql