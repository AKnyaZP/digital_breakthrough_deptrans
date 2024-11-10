package repository

import (
	pgdb "deptrans-backend/internal/pkg/postgres"
)

type GeoData interface {
}

type Repositories struct {
	GeoData
}

func NewRepositories(pg *pgdb.Postgres) *Repositories {
	return &Repositories{}
}
