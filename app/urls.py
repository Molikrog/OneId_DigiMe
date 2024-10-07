from django.contrib.auth.views import LogoutView
from django.urls import path

from . import views
from django.conf.urls.static import static
from django.conf import settings
from django.urls import include, path

from .views import id_check_compare

urlpatterns = [

    path('', views.home, name='home'),
    path('logout/', LogoutView.as_view(next_page = 'home'), name='logout'),
    path('register/', views.register_user, name='register'),
    path('compare_face/', views.compare_face, name='compare_face'),
    path('profile/', views.profile, name='profile'),
    path('documents/', views.documents, name='documents'),
    path('id_check/', views.id_check_view, name='id_check'),
    path('id_check_compare/', id_check_compare, name='id_check_compare'),
    ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
