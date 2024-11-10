package service

import (
	"deptrans-backend/internal/repository"
)

type GeoData interface {
}

type Services struct {
	GeoData
}

type ServicesDependencies struct {
	Repos *repository.Repositories
}

func NewServices(deps ServicesDependencies) *Services {
	return &Services{
		GeoData: NewMLService(deps.Repos.GeoData),
	}
}
