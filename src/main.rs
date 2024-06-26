use ash::ext::debug_utils;
use ash::ext::metal_surface;
use ash::khr::{swapchain, win32_surface};
use ash::prelude::VkResult;
use ash::vk::{
    self, DebugUtilsMessengerCreateInfoEXT, PhysicalDeviceFeatures2KHR,
    PhysicalDeviceVulkan13Features, PresentModeKHR, SurfaceCapabilitiesKHR, SurfaceFormatKHR,
    SurfaceKHR,
};
use glam::Mat4;
use glam::Vec3;
use image::buffer;
use image::DynamicImage;
use image::GenericImageView;
use image::ImageError;
use std::collections::HashSet;
use std::error::Error;
use std::f32::consts::PI;
use std::ffi::c_void;
use std::ffi::{c_char, CStr};
use std::fmt;
use std::fmt::Formatter;
use std::mem::offset_of;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;
use winit::window::WindowBuilder;
use winit::{
    event::{Event, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder, EventLoopWindowTarget},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle},
    window::Window,
};

// use ash::extensions::khr::surface;
use ash::khr::surface;

struct Clock {
    is_valid: bool, // 0 or 1
    previous_frame_instant: Instant,
    total_elapsed_time: Duration,
    delta_time: Duration,
    fps: u16,
}

impl Clock {
    pub fn tick(&mut self) {
        // A silly way to avoid calling tick more than once in the same frame. If the `Clock` would be globally available.
        // TODO change `is_valid` to a better name
        if self.is_valid {
            self.delta_time = self.previous_frame_instant.elapsed();
            self.previous_frame_instant = Instant::now();
            self.total_elapsed_time += self.delta_time;
            self.fps = (1000.0 / (self.delta_time.as_secs_f64() * 1000.0)) as u16;
        } else {
            warn!("Clock is not valid. It was ticked more than once in the same frame.");
        }
    }
}

impl std::fmt::Debug for Clock {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Clock")
            .field("FPS: ", &self.fps)
            .field("delta time: ", &self.delta_time)
            .field("total elapsed time: ", &self.total_elapsed_time)
            .finish()
    }
}

#[derive(Copy, Clone)]
// for this case having packed, c or both is the same. But I need to have them.
#[repr(C)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
    tex_coord: [f32; 2],
}

impl Vertex {
    fn get_binding_description() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)]
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, color) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Vertex, tex_coord) as u32),
        ]
    }
}

#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum WindowError {
    EventLoopError(#[from] winit::error::EventLoopError),
}

impl fmt::Display for WindowError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct RequestRedraw;
type UserEvent = RequestRedraw;

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let severity = format!("{:?}", message_severity).to_lowercase();
    let ty = format!("{:?}", message_type).to_lowercase();
    match severity.as_str() {
        "verbose" => {
            debug!("{} {:?}", ty, message);
        }
        "info" => {
            info!("{} {:?}", ty, message);
        }
        "warning" => {
            warn!("{} {:?}", ty, message);
        }
        "error" => {
            error!("{} {:?}", ty, message);
        }
        _ => (),
    }
    vk::FALSE
}

struct SwapChainSupportDetails {
    capabilities: SurfaceCapabilitiesKHR,
    formats: Vec<SurfaceFormatKHR>,
    present_modes: Vec<PresentModeKHR>,
}

impl SwapChainSupportDetails {
    pub fn new(
        physical_device: &vk::PhysicalDevice,
        surface: &SurfaceKHR,
        surface_loader: &surface::Instance,
    ) -> Self {
        let capabilities;
        let formats;
        let present_modes;

        unsafe {
            capabilities = surface_loader
                .get_physical_device_surface_capabilities(*physical_device, *surface)
                .unwrap();
            formats = surface_loader
                .get_physical_device_surface_formats(*physical_device, *surface)
                .unwrap();
            present_modes = surface_loader
                .get_physical_device_surface_present_modes(*physical_device, *surface)
                .unwrap()
        }

        Self {
            capabilities,
            formats,
            present_modes,
        }
    }
}

#[derive(Copy, Clone)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

#[cfg(target_os = "windows")]
const DEVICE_ENABLED_EXTENSION_NAMES: [*const c_char; 1] = [
    // ash::extensions::khr::swapchain::NAME.as_ptr()
    swapchain::NAME.as_ptr(),
];

#[cfg(any(target_os = "macos", target_os = "ios"))]
const DEVICE_ENABLED_EXTENSION_NAMES: [*const c_char; 2] = [
    // ash::extensions::khr::swapchain::NAME.as_ptr(),
    // ash::vk::khr::portability_subset::NAME.as_ptr(),
    ash::khr::swapchain::NAME.as_ptr(),
    ash::khr::portability_subset::NAME.as_ptr(),
];

const MODEL_PATH: &str = "src/models/viking_room.obj";
const TEXTURE_PATH: &str = "src/textures/viking_room.png";
enum BufferType {
    Vertex,
    Index,
}

#[derive(Debug)]
struct UniformBufferObject {
    model: glam::Mat4,
    view: glam::Mat4,
    proj: glam::Mat4,
}

struct VulkanApp {
    // window
    window: Arc<Window>,

    instance: ash::Instance,

    clock: Clock,

    // devices
    physical_device: vk::PhysicalDevice,
    physical_device_properties: vk::PhysicalDeviceProperties,
    device: ash::Device,

    // memory
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,

    // queue
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    // surface
    surface_loader: surface::Instance,
    surface: SurfaceKHR,

    // swapchain
    swapchain: vk::SwapchainKHR,
    swapchain_loader: swapchain::Device,
    swapchain_image_format: vk::Format,
    swapchain_extent: vk::Extent2D,

    swapchain_images: Vec<vk::Image>,
    swapchain_images_views: Vec<vk::ImageView>,

    // support
    queue_family_indices: QueueFamilyIndices,
    swapchain_support_details: SwapChainSupportDetails,

    // render pass
    render_pass: vk::RenderPass,

    // framebuffers
    swapchain_frame_buffers: Vec<vk::Framebuffer>,
    current_frame: usize,
    frame_buffer_resized: bool,

    //pipeline
    pipeline: vk::Pipeline,

    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    pipeline_layout: vk::PipelineLayout,

    // buffers
    // these two go together, probably should have them inside a struct and deallocated them together
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,

    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    // models
    model_vertices: Vec<Vertex>,
    // or maybe u16
    model_indices: Vec<u32>,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped: Vec<*mut c_void>,

    // command buffer
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    // textures
    mip_levels: u32,
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,

    // depth image and view
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,

    // sync objects
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,

    // debug
    debug_messenger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl VulkanApp {
    const MAX_FRAMES_IN_FLIGHT: usize = 2;

    // replaced by model
    // const VERTICES: [Vertex; 8] = [
    //     Vertex {
    //         pos: [-0.5, -0.5, 0.0],
    //         color: [1.0, 0.0, 0.0],
    //         tex_coord: [1.0, 0.0],
    //     },
    //     Vertex {
    //         pos: [0.5, -0.5, 0.0],
    //         color: [0.0, 1.0, 0.0],
    //         tex_coord: [0.0, 0.0],
    //     },
    //     Vertex {
    //         pos: [0.5, 0.5, 0.0],
    //         color: [0.0, 0.0, 1.0],
    //         tex_coord: [0.0, 1.0],
    //     },
    //     Vertex {
    //         pos: [-0.5, 0.5, 0.0],
    //         color: [1.0, 1.0, 1.0],
    //         tex_coord: [1.0, 1.0],
    //     },
    //     Vertex {
    //         pos: [-0.5, -0.5, -0.5],
    //         color: [1.0, 0.0, 0.0],
    //         tex_coord: [1.0, 0.0],
    //     },
    //     Vertex {
    //         pos: [0.5, -0.5, -0.5],
    //         color: [0.0, 1.0, 0.0],
    //         tex_coord: [0.0, 0.0],
    //     },
    //     Vertex {
    //         pos: [0.5, 0.5, -0.5],
    //         color: [0.0, 0.0, 1.0],
    //         tex_coord: [0.0, 1.0],
    //     },
    //     Vertex {
    //         pos: [-0.5, 0.5, -0.5],
    //         color: [1.0, 1.0, 1.0],
    //         tex_coord: [1.0, 1.0],
    //     },
    // ];

    // const INDICES: [u16; 12] = [0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4];
}

impl VulkanApp {
    pub unsafe fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) -> VkResult<vk::SurfaceKHR> {
        match (display_handle, window_handle) {
            (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(window)) => {
                let surface_desc = vk::Win32SurfaceCreateInfoKHR::default()
                    .hwnd(window.hwnd.get())
                    .hinstance(
                        window
                            .hinstance
                            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
                            .get(),
                    );
                let surface_fn = win32_surface::Instance::new(entry, instance);
                surface_fn.create_win32_surface(&surface_desc, allocation_callbacks)
            }
            #[cfg(target_os = "macos")]
            (RawDisplayHandle::AppKit(_), RawWindowHandle::AppKit(window)) => {
                use raw_window_metal::{appkit, Layer};

                let layer = match appkit::metal_layer_from_handle(window) {
                    Layer::Existing(layer) | Layer::Allocated(layer) => layer.cast(),
                };

                let surface_desc = vk::MetalSurfaceCreateInfoEXT::default().layer(&*layer);
                let surface_fn = metal_surface::Instance::new(entry, instance);
                surface_fn.create_metal_surface(&surface_desc, allocation_callbacks)
            }
            _ => unimplemented!(),
        }
    }

    fn create_instance(entry: &ash::Entry) -> Result<ash::Instance, String> {
        let app_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"VulkanTriangle\0") };

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(app_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let enabled_extension_names = &Self::required_extension_names(entry);

        // this is a bug, its being used...
        #[allow(unused_assignments)]
        let mut enabled_layer_names = vec![];

        #[allow(unused_assignments)]
        let mut debug_change_name: Option<DebugUtilsMessengerCreateInfoEXT> = None;

        #[cfg(debug_assertions)]
        {
            match Self::check_validation_layer_support(entry) {
                Ok(res) => {
                    enabled_layer_names = res;
                }
                Err(err) => {
                    return Err(err);
                }
            }
            debug_change_name = Some(Self::populate_debug_messenger_create_info());
        }

        #[allow(unused_mut)]
        let mut flags = vk::InstanceCreateFlags::empty();
        #[cfg(target_os = "macos")]
        {
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(enabled_extension_names)
            .enabled_layer_names(&enabled_layer_names)
            .flags(flags);

        let create_info = match debug_change_name {
            Some(ref mut info) => create_info.push_next(info),
            None => create_info,
        };

        unsafe {
            match entry.create_instance(&create_info, None) {
                Ok(instance) => Ok(instance),
                Err(err) => Err(format!("Failed to create instance: {:?}", err)),
            }
        }
    }

    fn create_physical_device(
        instance: &ash::Instance,
        surface_loader: &surface::Instance,
        surface: &SurfaceKHR,
    ) -> Result<
        (
            vk::PhysicalDevice,
            vk::PhysicalDeviceProperties,
            vk::PhysicalDeviceFeatures,
            vk::PhysicalDeviceVulkan12Features<'static>,
            PhysicalDeviceVulkan13Features<'static>,
            QueueFamilyIndices,
            SwapChainSupportDetails,
        ),
        String,
    > {
        unsafe {
            let physical_devices = instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices");

            if physical_devices.is_empty() {
                let err_msg = "No GPU with Vulkan support found!";
                error!(err_msg);
                return Err(err_msg.to_string());
            }

            let device_enabled_extension_names_as_c_strs = DEVICE_ENABLED_EXTENSION_NAMES
                .iter()
                .map(|&x| CStr::from_ptr(x))
                .collect::<Vec<_>>();

            for physical_device in physical_devices.iter() {
                let properties = instance.get_physical_device_properties(*physical_device);

                let physical_device_features =
                    instance.get_physical_device_features(*physical_device);

                // FIXME I'm enabling all the supported features for now. I should enable only the ones I need.
                // TODO This overwrites the `push_next` so I have no way, at the moment, of only enabling the features I want
                // Vulkan 1.2 Features
                let mut physical_device_12_features = vk::PhysicalDeviceVulkan12Features::default();
                let mut _12 = vk::PhysicalDeviceFeatures2KHR::default()
                    .push_next(&mut physical_device_12_features);
                instance.get_physical_device_features2(*physical_device, &mut _12);

                // Vulkan 1.3 Features
                // IMPORTANT:
                let mut physical_device_13_features = vk::PhysicalDeviceVulkan13Features::default();
                let mut _13 = PhysicalDeviceFeatures2KHR::default()
                    .push_next(&mut physical_device_13_features);
                // This line overwrites the `push_next`. So I should print `features2` after this line if I want to see
                // the features that are enabled for this GPU.
                instance.get_physical_device_features2(*physical_device, &mut _13);

                let device_name = CStr::from_ptr(properties.device_name.as_ptr())
                    .to_str()
                    .expect("Failed to get device name");
                let device_type = properties.device_type;

                // verifies if the device has the required extensions. At the moment only swapchain
                let mut required_extensions = HashSet::new();
                required_extensions.extend(device_enabled_extension_names_as_c_strs.clone());

                let available_extensions = instance
                    .enumerate_device_extension_properties(*physical_device)
                    .unwrap();

                debug!("Available extensions for device: {:?}", device_name);
                for extension in available_extensions.iter() {
                    debug!("{:?}", &extension.extension_name_as_c_str().unwrap());
                    required_extensions.remove(&extension.extension_name_as_c_str().unwrap());
                }

                #[cfg(debug_assertions)]
                {
                    let available_layers = instance
                        .enumerate_device_layer_properties(*physical_device)
                        .unwrap();
                    debug!("These layers are available for the device: ");
                    for available_layer in available_layers {
                        debug!(
                            "{:?}",
                            std::ffi::CStr::from_ptr(available_layer.layer_name.as_ptr())
                        );
                    }
                }
                // verifies if the device has the required extensions. At the moment only swapchain

                // this is `isDeviceSuitable` in C++
                if required_extensions.is_empty() {
                    let swapchain_support_details =
                        SwapChainSupportDetails::new(physical_device, surface, surface_loader);
                    if swapchain_support_details.formats.is_empty()
                        || swapchain_support_details.present_modes.is_empty()
                    {
                        continue;
                    }
                    info!(
                        "Physical Device (GPU): {}, Device Type: {:?}, Extensions: {:?}",
                        device_name, device_type, device_enabled_extension_names_as_c_strs
                    );
                    let queue_families_indices = Self::find_queue_families(
                        instance,
                        physical_device,
                        surface_loader,
                        surface,
                    );
                    return Ok((
                        *physical_device,
                        properties,
                        physical_device_features,
                        physical_device_12_features,
                        physical_device_13_features,
                        queue_families_indices,
                        swapchain_support_details,
                    ));
                }
            }
        }

        let err_msg = "No suitable GPU found!".to_string();
        error!("{:?}", err_msg);
        Err(err_msg)
    }

    fn find_queue_families(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        surface_loader: &surface::Instance,
        surface: &SurfaceKHR,
    ) -> QueueFamilyIndices {
        unsafe {
            let queue_families =
                instance.get_physical_device_queue_family_properties(*physical_device);
            let mut indices = QueueFamilyIndices {
                graphics_family: None,
                present_family: None,
            };

            for (i, queue_family) in queue_families.iter().enumerate() {
                if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    indices.graphics_family = Some(i as u32);
                }

                let is_present_support = surface_loader
                    .get_physical_device_surface_support(*physical_device, i as u32, *surface)
                    .unwrap();

                if is_present_support {
                    indices.present_family = Some(i as u32);
                }

                if indices.is_complete() {
                    break;
                }
            }
            indices
        }
    }

    fn create_image(
        device: &ash::Device,
        tex_width: u32,
        tex_height: u32,
        mip_levels: u32,
        format: vk::Format,
        image_tiling: vk::ImageTiling,
        image_usages: vk::ImageUsageFlags,
        memory_prop: &vk::PhysicalDeviceMemoryProperties,
        memory_flags: vk::MemoryPropertyFlags,
    ) -> VkResult<(vk::Image, vk::DeviceMemory)> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(
                vk::Extent3D::default()
                    .width(tex_width)
                    .height(tex_height)
                    .depth(1),
            )
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(image_tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(image_usages)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);

        unsafe {
            let texture_image = match device.create_image(&image_info, None) {
                Ok(texture_image) => texture_image,
                Err(err) => {
                    let err_msg = format!("Failed to create texture image: {:?}", err);
                    error!("{:?}", err_msg);
                    panic!("{:?}", err_msg);
                }
            };

            let mem_requirements = device.get_image_memory_requirements(texture_image);
            let memory_type_index: Option<u32> =
                Self::find_memorytype_index(&mem_requirements, memory_prop, memory_flags);

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(memory_type_index.unwrap());

            // It should be noted that in a real world application, you're not supposed to actually call
            // vkAllocateMemory for every individual buffer.
            let memory = match device.allocate_memory(&alloc_info, None) {
                Ok(memory) => {
                    // TODO
                    // lo saco para afuera, porque esta funcion va a formar parte de una funcion de alocacion y segun el error que tira podemos
                    // decir si hubo error creando un vertex buffer memory or un texture image memory etc

                    unsafe { device.bind_image_memory(texture_image, memory, 0) }.unwrap();
                    memory
                }
                Err(err) => {
                    let err_msg = format!("Failed to allocate image memory: {:?}", err);
                    error!("{:?}", err_msg);
                    panic!("{:?}", err_msg);
                }
            };
            Ok((texture_image, memory))
        }
    }

    fn create_texture_sampler(device: &ash::Device, mip_levels: u32) -> VkResult<vk::Sampler> {
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(16.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .min_lod(0.0)
            .max_lod(mip_levels as f32)
            .mip_lod_bias(0.0);

        unsafe {
            match device.create_sampler(&sampler_info, None) {
                Ok(sampler) => Ok(sampler),
                Err(err) => {
                    let err_msg = format!("Failed to create texture sampler: {:?}", err);
                    error!("{:?}", err_msg);
                    Err(err)
                }
            }
        }
    }

    fn create_texture_image_view(
        device: &ash::Device,
        texture_image: vk::Image,
        format: vk::Format,
        mip_levels: u32,
    ) -> VkResult<vk::ImageView> {
        return Self::create_image_view(
            device,
            texture_image,
            format,
            mip_levels,
            vk::ImageAspectFlags::COLOR,
        );
    }

    fn create_image_view(
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
        mip_levels: u32,
        aspect_flags: vk::ImageAspectFlags,
    ) -> VkResult<vk::ImageView> {
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            })
            // Referenced here: https://vkguide.dev/docs/new_chapter_1/vulkan_mainloop_code/
            // .subresource_range(
            //     vk::ImageSubresourceRange::default()
            //         .aspect_mask(vk::ImageAspectFlags::DEPTH or vk::ImageAspectFlags::color) based on
            // VK_IMAGE_LAYOUR_DEPTH_ATTACHMENT relative to the tutorial
            //         .base_mip_level(0)
            //         .level_count(vk::REMAINING_MIP_LEVELS)
            //         .base_array_layer(0)
            //         .layer_count(vk::REMAINING_ARRAY_LAYERS),
            // );
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_flags)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        match unsafe { device.create_image_view(&create_info, None) } {
            Ok(view) => Ok(view),
            Err(e) => {
                error!("Failed to create image view: {:?}", e);
                return Err(e);
            }
        }
    }

    fn create_depth_resources(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        command_pool: &vk::CommandPool,
        graphics_queue: &vk::Queue,
        swapchain_extent: &vk::Extent2D,
        memory_prop: &vk::PhysicalDeviceMemoryProperties,
    ) -> VkResult<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
        let depth_format = Self::find_depth_format(instance, physical_device);
        let (depth_image, depth_image_memory) = Self::create_image(
            device,
            swapchain_extent.width,
            swapchain_extent.height,
            1,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            memory_prop,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .unwrap();
        let depth_image_view = Self::create_image_view(
            device,
            depth_image,
            depth_format,
            1,
            vk::ImageAspectFlags::DEPTH,
        )
        .unwrap();

        // this is not needed since we do it in the renderpass, but for completeness:
        Self::transition_image_layout(
            device,
            command_pool,
            graphics_queue,
            depth_image,
            depth_format,
            1,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );

        Ok((depth_image, depth_image_memory, depth_image_view))
    }

    fn find_depth_format(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
    ) -> vk::Format {
        return Self::find_supported_format(
            instance,
            physical_device,
            &[
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        );
    }

    fn find_supported_format(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> vk::Format {
        for &format in candidates.iter() {
            let props =
                unsafe { instance.get_physical_device_format_properties(*physical_device, format) };
            if tiling == vk::ImageTiling::LINEAR && props.linear_tiling_features.contains(features)
            {
                return format;
            } else if tiling == vk::ImageTiling::OPTIMAL
                && props.optimal_tiling_features.contains(features)
            {
                return format;
            }
        }
        error!("Failed to find supported formats: {:?}", candidates);
        panic!("Failed to find supported formats: {:?}", candidates);
    }

    fn has_stencil_component(format: vk::Format) -> bool {
        format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
    }

    fn create_texture_image(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        command_pool: &vk::CommandPool,
        graphics_queue: &vk::Queue,
        format: vk::Format,
        memory_prop: &vk::PhysicalDeviceMemoryProperties,
    ) -> VkResult<(vk::Image, vk::DeviceMemory, u32)> {
        // replaced by model
        // let image = match image::open("src/textures/texture.jpg") {
        let image = match image::open("src/textures/viking_room.png") {
            Ok(image) => image,
            Err(e) => {
                error!("Failed to load image: {:?}", e);
                panic!("Failed to load image: {:?}", e);
            }
        };

        let image_as_rgb = image.to_rgba8();
        let image_width = (&image_as_rgb).width();
        let image_height = (&image_as_rgb).height();
        let pixels = image_as_rgb.into_raw();

        // both of these calculations give the same result. At least for now... I'm not sure which one to choose.
        let image_size2 = (pixels.len() * std::mem::size_of::<u8>()) as vk::DeviceSize;
        let image_size = vk::DeviceSize::from(image.width() * image.height() * 4);
        let mip_levels = ((image_width.max(image_height) as f32).log2().floor() + 1.0) as u32;

        // ----------------- TODO SCAR DE ACA -----------------------
        // se comparte en todas las creaciones de staging buffers

        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            device,
            image_size2,
            vk::BufferUsageFlags::TRANSFER_SRC,
            memory_prop,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();

        unsafe {
            let data = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    image_size2,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();
            // let mut align =
            //     ash::util::Align::new(data, std::mem::align_of::<u8>() as u64, image_size);
            let mut align =
                ash::util::Align::new(data, std::mem::align_of::<u8>() as u64, image_size2);
            // align.copy_from_slice(&image.as_bytes());
            align.copy_from_slice(&pixels);
            device.unmap_memory(staging_buffer_memory);
        }

        // ----------------- TODO SCAR DE ACA -----------------------

        let (texture_image, texture_image_memory) = Self::create_image(
            device,
            // image.width(),
            // image.height(),
            image_width,
            image_height,
            mip_levels,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            memory_prop,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .unwrap();

        Self::transition_image_layout(
            device,
            command_pool,
            graphics_queue,
            texture_image,
            format,
            mip_levels,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        Self::copy_buffer_to_image(
            device,
            command_pool,
            graphics_queue,
            staging_buffer,
            texture_image,
            // image.width(),
            // image.height(),
            image_width,
            image_height,
        );

        // removed this on miplevel chapter. because`transition_image_layout` only performs layout transitions
        // on the entire image. So after generating the mipmaps each level will transition to `SHADER_READ_ONLY_OPTIMAL`
        // Self::transition_image_layout(
        //     device,
        //     command_pool,
        //     graphics_queue,
        //     texture_image,
        //     format,
        //     mip_levels,
        //     vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        //     vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        // );

        Self::generate_mipmaps(
            device,
            instance,
            physical_device,
            command_pool,
            graphics_queue,
            texture_image,
            format,
            image_width,
            image_height,
            mip_levels,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        Ok((texture_image, texture_image_memory, mip_levels))
    }

    fn generate_mipmaps(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        command_pool: &vk::CommandPool,
        graphics_queue: &vk::Queue,
        image: vk::Image,
        format: vk::Format,
        tex_width: u32,
        tex_height: u32,
        mip_levels: u32,
    ) {
        // It should be noted that generating mipmaps at runtime is an uncommon practice.
        // They should be generated in software and stored in the texture file.
        Self::find_supported_format(
            instance,
            physical_device,
            &[format],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
        );

        let command_buffer = Self::begin_single_time_commands(device, command_pool);
        let mut image_memory_barrier = vk::ImageMemoryBarrier::default()
            .image(image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .level_count(1),
            );

        let mut mip_width = tex_width as i32;
        let mut mip_height = tex_height as i32;

        for i in 1..mip_levels {
            image_memory_barrier.subresource_range.base_mip_level = i - 1;
            image_memory_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            image_memory_barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            image_memory_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            image_memory_barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_memory_barrier.clone()],
                );
            }

            let blits = [vk::ImageBlit::default()
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ])
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(i - 1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: if mip_width > 1 { mip_width / 2 } else { 1 },
                        y: if mip_height > 1 { mip_height / 2 } else { 1 },
                        z: 1,
                    },
                ])
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(i)
                        .base_array_layer(0)
                        .layer_count(1),
                )];

            unsafe {
                device.cmd_blit_image(
                    command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &blits,
                    vk::Filter::LINEAR,
                );

                image_memory_barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                image_memory_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                image_memory_barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
                image_memory_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_memory_barrier.clone()],
                );
            }

            if mip_width > 1 {
                mip_width /= 2;
            }
            if mip_height > 1 {
                mip_height /= 2;
            }
        }

        unsafe {
            image_memory_barrier.subresource_range.base_mip_level = mip_levels - 1;
            image_memory_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            image_memory_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            image_memory_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            image_memory_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_memory_barrier.clone()],
            );
        }
        Self::end_single_time_commands(device, &command_buffer, command_pool, graphics_queue);
    }

    fn copy_buffer_to_image(
        device: &ash::Device,
        command_pool: &vk::CommandPool,
        graphics_queue: &vk::Queue,
        buffer: vk::Buffer,
        image: vk::Image,
        width: u32,
        height: u32,
    ) {
        let command_buffer = Self::begin_single_time_commands(device, command_pool);
        let command_buffers = [command_buffer];

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        unsafe {
            device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        Self::end_single_time_commands(device, &command_buffer, command_pool, graphics_queue);
    }

    fn begin_single_time_commands(
        device: &ash::Device,
        command_pool: &vk::CommandPool,
    ) -> vk::CommandBuffer {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe { device.allocate_command_buffers(&allocate_info).unwrap()[0] };
        unsafe {
            device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }
        command_buffer
    }

    fn end_single_time_commands(
        device: &ash::Device,
        command_buffer: &vk::CommandBuffer,
        command_pool: &vk::CommandPool,
        graphics_queue: &vk::Queue,
    ) {
        unsafe {
            device.end_command_buffer(*command_buffer).unwrap();

            let command_buffers = [*command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

            device
                .queue_submit(*graphics_queue, &[submit_info], vk::Fence::null())
                .expect("Failed to submit queue");

            device.queue_wait_idle(*graphics_queue).unwrap();
            device.free_command_buffers(*command_pool, &command_buffers)
        }
    }

    fn create_swapchain(
        instance: &ash::Instance,
        device: &ash::Device,
        surface: &SurfaceKHR,
        swapchain_support_details: &SwapChainSupportDetails,
        window_size: (u32, u32),
        queue_family_indices: &QueueFamilyIndices,
    ) -> VkResult<(
        vk::SwapchainKHR,
        swapchain::Device,
        Vec<vk::Image>,
        vk::Format,
        vk::Extent2D,
    )> {
        let swapchain_image_format: SurfaceFormatKHR = *swapchain_support_details
            .formats
            .iter()
            .find(|format| {
                // TODO Why using B8G8R8A8_SRGB does not merge colors as B8G8R8A8_UNORM does
                format.format == vk::Format::R8G8B8A8_SRGB
                // format.format == vk::Format::B8G8R8A8_UNORM
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&swapchain_support_details.formats[0]);

        let present_mode = swapchain_support_details
            .present_modes
            .iter()
            .find(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(&vk::PresentModeKHR::FIFO);

        let extent: vk::Extent2D = {
            if swapchain_support_details.capabilities.current_extent.width != u32::MAX {
                swapchain_support_details.capabilities.current_extent
            } else {
                vk::Extent2D {
                    width: window_size.0.clamp(
                        swapchain_support_details
                            .capabilities
                            .min_image_extent
                            .width,
                        swapchain_support_details
                            .capabilities
                            .max_image_extent
                            .width,
                    ),
                    height: window_size.1.clamp(
                        swapchain_support_details
                            .capabilities
                            .min_image_extent
                            .height,
                        swapchain_support_details
                            .capabilities
                            .max_image_extent
                            .height,
                    ),
                }
            }
        };

        let image_count = swapchain_support_details.capabilities.min_image_count + 1;

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(*surface)
            .min_image_count(image_count)
            .image_format(swapchain_image_format.format)
            .image_color_space(swapchain_image_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(swapchain_support_details.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(*present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        let indices = &[
            queue_family_indices.graphics_family.unwrap(),
            queue_family_indices.present_family.unwrap(),
        ];
        let swapchain_create_info =
            match queue_family_indices.graphics_family != queue_family_indices.present_family {
                true => {
                    /*
                    "If the queue families differ, then we'll be using the concurrent mode in this tutorial to avoid
                    having to do the ownership chapters, because these involve some concepts that are better explained
                    at a later time. Concurrent mode requires you to specify in advance between which queue families
                    ownership will be shared using the queueFamilyIndexCount and pQueueFamilyIndices parameters.
                    If the graphics queue family and presentation queue family are the same, which will be the case on
                    most hardware, then we should stick to exclusive mode, because concurrent mode requires you to
                    specify at least two distinct queue families."
                    - https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain
                    */
                    swapchain_create_info
                        .image_sharing_mode(vk::SharingMode::CONCURRENT)
                        .queue_family_indices(indices)
                }
                false => swapchain_create_info
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .queue_family_indices(&[]),
            };

        unsafe {
            let swapchain_loader = swapchain::Device::new(instance, device);
            match swapchain_loader.create_swapchain(&swapchain_create_info, None) {
                Ok(swapchain) => {
                    info!("Swapchain created successfully");
                    match swapchain_loader.get_swapchain_images(swapchain) {
                        Ok(swapchain_images) => {
                            info!("Swapchain Images obtained successfully");
                            Ok((
                                swapchain,
                                swapchain_loader,
                                swapchain_images,
                                swapchain_image_format.format,
                                extent,
                            ))
                        }
                        Err(err) => {
                            let err_msg = format!("Failed to get swapchain images: {:?}", err);
                            error!("{:?}", err_msg);
                            Err(err)
                        }
                    }
                }
                Err(err) => {
                    let err_msg = format!("Failed to create swapchain: {:?}", err);
                    error!("{:?}", err_msg);
                    Err(err)
                }
            }
        }
    }

    fn create_swapchain_images_views(
        device: &ash::Device,
        swapchain_images: &Vec<vk::Image>,
        swapchain_image_format: vk::Format,
    ) -> VkResult<Vec<vk::ImageView>> {
        let mut image_views = Vec::with_capacity(swapchain_images.len());

        for image in swapchain_images {
            match Self::create_image_view(
                device,
                *image,
                swapchain_image_format,
                1,
                vk::ImageAspectFlags::COLOR,
            ) {
                Ok(view) => image_views.push(view),
                Err(e) => return Err(e),
            }
        }

        Ok(image_views)
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        physical_device_features: vk::PhysicalDeviceFeatures,
        mut physical_device_12_features: vk::PhysicalDeviceVulkan12Features,
        mut physical_device_13_features: PhysicalDeviceVulkan13Features,
        indices: QueueFamilyIndices,
    ) -> Result<ash::Device, String> {
        let mut unique_queue_families = HashSet::new();
        unique_queue_families.insert(indices.graphics_family.unwrap());
        unique_queue_families.insert(indices.present_family.unwrap());

        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo<'static>> = unique_queue_families
            .iter()
            .map(|queue_family_index| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*queue_family_index)
                    .queue_priorities(&[1.0])
            })
            .collect();

        // let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true).descriptor_indexing(true);
        // let mut vulkan_13_features = PhysicalDeviceVulkan13Features::default().dynamic_rendering(true).synchronization2(true);
        // let mut vulkan_13_features = PhysicalDeviceVulkan13Features::default().dynamic_rendering(true).synchronization2(true).texture_compression_astc_hdr(true);

        println!("---------------------------------\n");
        println!("Device Features {:?}\n", physical_device_features);
        println!("Device Features 1.2 {:?}\n", physical_device_12_features);
        println!("Device Features 1.3 {:?}\n", physical_device_13_features);

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&DEVICE_ENABLED_EXTENSION_NAMES)
            .enabled_features(&physical_device_features)
            .push_next(&mut physical_device_12_features)
            .push_next(&mut physical_device_13_features);

        unsafe {
            match instance.create_device(*physical_device, &create_info, None) {
                Ok(device) => Ok(device),
                Err(err) => {
                    let err_msg = format!("Failed to create logical device: {:?}", err);
                    error!("{:?}", err_msg);
                    Err(err_msg)
                }
            }
        }
    }

    pub fn new(window: Arc<Window>) -> Result<Self, String> {
        let clock = Clock {
            is_valid: true,
            previous_frame_instant: Instant::now(),
            total_elapsed_time: Duration::default(),
            delta_time: Duration::default(),
            fps: 0,
        };

        // let entry = unsafe { ash::Entry::load().unwrap() };
        let entry = ash::Entry::linked();

        let instance = Self::create_instance(&entry)?;
        let debug_messenger = Self::setup_debug_messenger(&entry, &instance);

        let display_handle = window.display_handle().unwrap().as_raw();
        let window_handle = window.window_handle().unwrap().as_raw();

        // por ahora `surface` y `surface_loader` siempre se pasan juntas a todos los metodos que necesitan una
        let surface: SurfaceKHR = unsafe {
            Self::create_surface(&entry, &instance, display_handle, window_handle, None).unwrap()
        };
        let surface_loader: surface::Instance = surface::Instance::new(&entry, &instance);

        let (
            physical_device,
            physical_device_properties,
            physical_device_features,
            physical_device_12_features,
            physical_device_13_features,
            queue_family_indices,
            swapchain_support_details,
        ) = Self::create_physical_device(&instance, &surface_loader, &surface)?;

        // TODO analyze if this should be inside `create_physical_device`
        let device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let device = Self::create_logical_device(
            &instance,
            &physical_device,
            physical_device_features,
            physical_device_12_features,
            physical_device_13_features,
            queue_family_indices,
        )?;

        let graphics_queue =
            unsafe { device.get_device_queue(queue_family_indices.graphics_family.unwrap(), 0) };
        let present_queue =
            unsafe { device.get_device_queue(queue_family_indices.present_family.unwrap(), 0) };

        debug!("Graphics Queue: {:?}", graphics_queue);
        debug!("Present Queue: {:?}", present_queue);
        debug!("They are the same: {:?}", graphics_queue == present_queue);

        let (
            swapchain,
            swapchain_loader,
            swapchain_images,
            swapchain_image_format,
            swapchain_extent,
        ) = Self::create_swapchain(
            &instance,
            &device,
            &surface,
            &swapchain_support_details,
            window.inner_size().into(),
            &queue_family_indices,
        )
        .unwrap();

        let swapchain_images_views =
            Self::create_swapchain_images_views(&device, &swapchain_images, swapchain_image_format)
                .unwrap();

        let render_pass = Self::create_render_pass(
            &device,
            &instance,
            &physical_device,
            &swapchain_image_format,
        )
        .unwrap();

        let descriptor_set_layout = Self::create_descriptor_set_layout(&device).unwrap();

        let (pipeline, pipeline_layout) = Self::create_graphics_pipeline(
            &device,
            &swapchain_extent,
            &render_pass,
            &[descriptor_set_layout],
        )
        .unwrap();

        let command_pool = Self::create_command_pool(&device, &queue_family_indices).unwrap();

        let (depth_image, depth_image_memory, depth_image_view) = Self::create_depth_resources(
            &device,
            &instance,
            &physical_device,
            &command_pool,
            &graphics_queue,
            &swapchain_extent,
            &device_memory_properties,
        )
        .unwrap();

        let swapchain_frame_buffers = Self::create_frame_buffers(
            &device,
            &swapchain_images_views,
            &render_pass,
            &swapchain_extent,
            depth_image_view,
        )
        .unwrap();

        let image_format = vk::Format::R8G8B8A8_SRGB;
        let (texture_image, texture_image_memory, mip_levels) = Self::create_texture_image(
            &device,
            &instance,
            &physical_device,
            &command_pool,
            &graphics_queue,
            image_format,
            &device_memory_properties,
        )
        .unwrap();

        let texture_image_view =
            Self::create_texture_image_view(&device, texture_image, image_format, mip_levels)
                .unwrap();
        info!(
            "Max sampler anisotropy: {}",
            physical_device_properties.limits.max_sampler_anisotropy
        );
        let texture_sampler = Self::create_texture_sampler(&device, mip_levels).unwrap();

        let (model_vertices, model_indices) = Self::load_model();

        let (vertex_buffer, vertex_buffer_memory) = Self::create_buffers(
            &device,
            &command_pool,
            &graphics_queue,
            &device_memory_properties,
            // Self::VERTICES.as_ptr(),
            model_vertices.as_ptr(),
            BufferType::Vertex,
            // std::mem::size_of_val(&Self::VERTICES) as u64,
            std::mem::size_of::<Vertex>() as u64 * model_vertices.len() as u64,
        )
        .unwrap();

        let (index_buffer, index_buffer_memory) = Self::create_buffers(
            &device,
            &command_pool,
            &graphics_queue,
            &device_memory_properties,
            // Self::INDICES.as_ptr(),
            model_indices.as_ptr(),
            BufferType::Index,
            // std::mem::size_of_val(&Self::INDICES) as u64,
            std::mem::size_of::<u32>() as u64 * model_indices.len() as u64,
        )
        .unwrap();

        let (uniform_buffers, uniform_buffers_memory, uniform_buffers_mapped) =
            Self::create_uniform_buffers(
                &device,
                &command_pool,
                &graphics_queue,
                &device_memory_properties,
                std::mem::size_of::<UniformBufferObject>() as u64,
                // std::mem::size_of_val(&Self::INDICES) as u64,
            )
            .unwrap();
        let descriptor_pool = Self::create_descriptor_pool(&device).unwrap();
        let descriptor_sets = Self::create_descriptor_sets(
            &device,
            descriptor_set_layout,
            descriptor_pool,
            &uniform_buffers,
            texture_image_view,
            texture_sampler,
        )
        .unwrap();

        let command_buffers = Self::create_command_buffers(&device, &command_pool).unwrap();

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            Self::create_sync_objects(&device).unwrap();

        Ok(Self {
            window,
            clock,

            instance,
            debug_messenger,
            physical_device,
            physical_device_properties,
            device,

            graphics_queue,
            present_queue,

            surface,
            surface_loader,

            device_memory_properties,

            swapchain_image_format,
            swapchain_extent,
            swapchain,
            swapchain_loader,
            swapchain_images,
            swapchain_images_views,
            current_frame: 0,
            frame_buffer_resized: false,

            render_pass,

            swapchain_frame_buffers,

            queue_family_indices,
            swapchain_support_details,

            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,

            pipeline_layout,
            pipeline,

            // buffers
            vertex_buffer,
            vertex_buffer_memory,

            index_buffer,
            index_buffer_memory,

            uniform_buffers,
            uniform_buffers_memory,
            uniform_buffers_mapped,

            command_pool,
            command_buffers,

            mip_levels,
            texture_image,
            texture_image_memory,
            texture_image_view,
            texture_sampler,

            // depth image and view
            depth_image,
            depth_image_memory,
            depth_image_view,

            // models
            model_vertices,
            model_indices,

            // sync objects
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        })
    }

    fn create_buffer(
        device: &ash::Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_prop: &vk::PhysicalDeviceMemoryProperties,
        memory_flags: vk::MemoryPropertyFlags,
    ) -> VkResult<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = match unsafe { device.create_buffer(&buffer_info, None) } {
            Ok(buffer) => buffer,
            Err(err) => {
                error!("Failed to create vertex buffer: {:?}", err);
                panic!()
            }
        };

        // ------------------ esto me lo puedo llevar a un modulo -----------------
        // ------------------------- TODO -------------------------
        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type_index: Option<u32> =
            Self::find_memorytype_index(&mem_requirements, memory_prop, memory_flags);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index.unwrap());

        // It should be noted that in a real world application, you're not supposed to actually call
        // vkAllocateMemory for every individual buffer.
        let buffer_memory = match unsafe { device.allocate_memory(&alloc_info, None) } {
            Ok(buffer_memory) => {
                // TODO
                // lo saco para afuera, porque esta funcion va a formar parte de una funcion de alocacion y segun el error que tira podemos
                // decir si hubo error creando un vertex buffer memory or un texture image memory etc

                unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0) }.unwrap();
                buffer_memory
            }
            Err(err) => {
                let err_msg = format!("Failed to allocate vertex buffer memory: {:?}", err);
                error!("{:?}", err_msg);
                panic!("{:?}", err_msg);
            }
        };
        // ------------------------- TODO -------------------------
        // ------------------ esto me lo puedo llevar a un modulo -----------------

        Ok((buffer, buffer_memory))
    }

    fn create_descriptor_pool(device: &ash::Device) -> VkResult<vk::DescriptorPool> {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(Self::MAX_FRAMES_IN_FLIGHT as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(Self::MAX_FRAMES_IN_FLIGHT as u32),
        ];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(Self::MAX_FRAMES_IN_FLIGHT as u32);

        unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None) }
    }

    fn create_descriptor_sets(
        device: &ash::Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
        descriptor_pool: vk::DescriptorPool,
        uniform_buffers: &Vec<vk::Buffer>,
        texture_image_view: vk::ImageView,
        texture_sampler: vk::Sampler,
    ) -> VkResult<Vec<vk::DescriptorSet>> {
        // In our case we will create one descriptor set for each frame in flight, all with the same layout.
        // Unfortunately we do need all the copies of the layout because the next function expects an array matching the number of sets.
        let layouts = vec![descriptor_set_layout; Self::MAX_FRAMES_IN_FLIGHT];

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .unwrap()
        };

        for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
            let buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers[i])
                .offset(0)
                .range(std::mem::size_of::<UniformBufferObject>() as u64)];

            let image_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(texture_image_view)
                .sampler(texture_sampler)];

            let descriptor_write = [
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_sets[i])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_sets[i])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(image_info.as_slice()),
            ];

            unsafe {
                device.update_descriptor_sets(descriptor_write.as_slice(), &[]);
            }
        }
        Ok(descriptor_sets)
    }

    fn create_uniform_buffers(
        device: &ash::Device,
        command_pool: &vk::CommandPool,
        graphics_queue: &vk::Queue,
        memory_prop: &vk::PhysicalDeviceMemoryProperties,
        // buffer_data: *const T,
        // buffer_type: BufferType,
        buffer_size: vk::DeviceSize,
    ) -> VkResult<(Vec<vk::Buffer>, Vec<vk::DeviceMemory>, Vec<*mut c_void>)> {
        let mut uniform_buffers = Vec::with_capacity(Self::MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers_memory = Vec::with_capacity(Self::MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers_mapped = Vec::with_capacity(Self::MAX_FRAMES_IN_FLIGHT);
        for _ in 0..Self::MAX_FRAMES_IN_FLIGHT {
            let (uniform_buffer, uniform_buffer_memory) = Self::create_buffer(
                device,
                buffer_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                memory_prop,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .unwrap();
            uniform_buffers.push(uniform_buffer);
            uniform_buffers_memory.push(uniform_buffer_memory);
            // We map the buffer right after creation using vkMapMemory to get a pointer to which we can write the data later on.
            // The buffer stays mapped to this pointer for the application's whole lifetime.
            // This technique is called "persistent mapping" and works on all Vulkan implementations.
            // Not having to map the buffer every time we need to update it increases performances, as mapping is not free.
            uniform_buffers_mapped.push(unsafe {
                device
                    .map_memory(
                        uniform_buffer_memory,
                        0,
                        buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .unwrap()
            });
        }
        Ok((
            uniform_buffers,
            uniform_buffers_memory,
            uniform_buffers_mapped,
        ))
    }

    fn load_model() -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = vec![];
        let mut indices = vec![];
        let model = tobj::load_obj("src/models/viking_room.obj", &tobj::LoadOptions::default());
        let (models, _materials) = model.unwrap();
        for m in models.iter() {
            let mesh = &m.mesh;
            for f in 0..mesh.indices.len() {
                let vertex = Vertex {
                    pos: [
                        mesh.positions[mesh.indices[f] as usize * 3],
                        mesh.positions[mesh.indices[f] as usize * 3 + 1],
                        mesh.positions[mesh.indices[f] as usize * 3 + 2],
                    ],
                    tex_coord: [
                        mesh.texcoords[mesh.indices[f] as usize * 2],
                        1.0 - mesh.texcoords[mesh.indices[f] as usize * 2 + 1],
                    ],
                    color: [1.0, 1.0, 1.0],
                };
                vertices.push(vertex);
                indices.push(f as u32);
            }
        }
        (vertices, indices)
    }

    fn create_buffers<T>(
        device: &ash::Device,
        command_pool: &vk::CommandPool,
        graphics_queue: &vk::Queue,
        memory_prop: &vk::PhysicalDeviceMemoryProperties,
        buffer_data: *const T,
        buffer_type: BufferType,
        buffer_size: vk::DeviceSize,
    ) -> VkResult<(vk::Buffer, vk::DeviceMemory)> {
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            memory_prop,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();

        unsafe {
            // TODO Do error handling
            let data = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();
            std::ptr::copy_nonoverlapping(
                buffer_data as *const u8,
                data as *mut u8,
                buffer_size as usize,
            );
            // this works as well
            // let mut align = ash::util::Align::new(
            //     data,
            //     std::mem::align_of::<u32>() as _,
            //     mem_requirements.size as _,
            // );
            // align.copy_from_slice(&vertices);
            device.unmap_memory(staging_buffer_memory);
        }

        // TODO se crea pero no se unmappea. por que?
        // TODO y por que antes tenia que unmappearlo? tiene que ver con que uno queda en el cpu
        let (buffer, buffer_memory) = match buffer_type {
            BufferType::Vertex => Self::create_buffer(
                device,
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                memory_prop,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .unwrap(),
            BufferType::Index => Self::create_buffer(
                device,
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                memory_prop,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .unwrap(),
        };

        Self::copy_buffer(
            device,
            command_pool,
            graphics_queue,
            staging_buffer,
            buffer,
            buffer_size,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        Ok((buffer, buffer_memory))
    }

    fn copy_buffer(
        device: &ash::Device,
        command_pool: &vk::CommandPool,
        graphics_queue: &vk::Queue,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        size: vk::DeviceSize,
    ) {
        // let allocate_info = vk::CommandBufferAllocateInfo::default()
        //     .command_pool(*command_pool)
        //     .level(vk::CommandBufferLevel::PRIMARY)
        //     .command_buffer_count(1);

        // let command_buffer = unsafe { device.allocate_command_buffers(&allocate_info).unwrap()[0] };
        unsafe {
            let command_buffer = Self::begin_single_time_commands(device, command_pool);
            // device
            //     .begin_command_buffer(
            //         command_buffer,
            //         &vk::CommandBufferBeginInfo::default()
            //             .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            //     )
            //     .unwrap();

            let regions = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(size);
            device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &[regions]);

            Self::end_single_time_commands(device, &command_buffer, command_pool, graphics_queue)

            // device
            //     .end_command_buffer(command_buffer)
            //     .expect("Failed to record command buffer");

            // let command_buffers = [command_buffer];
            // let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
            // device
            //     .queue_submit(*graphics_queue, &[submit_info], vk::Fence::null())
            //     .expect("Failed to submit queue");

            // device.queue_wait_idle(*graphics_queue).unwrap();
            // device.free_command_buffers(*command_pool, &command_buffers)
        };
    }

    fn transition_image_layout(
        device: &ash::Device,
        command_pool: &vk::CommandPool,
        graphics_queue: &vk::Queue,
        image: vk::Image,
        format: vk::Format,
        mip_levels: u32,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let command_buffer = Self::begin_single_time_commands(device, command_pool);

        let mut image_barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
            image_barrier.subresource_range.aspect_mask = vk::ImageAspectFlags::DEPTH;

            if Self::has_stencil_component(format) {
                image_barrier.subresource_range.aspect_mask |= vk::ImageAspectFlags::STENCIL;
            }
        }

        let source_stage;
        let destination_stage;

        if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            image_barrier.src_access_mask = vk::AccessFlags::empty();
            image_barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;

            source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
            destination_stage = vk::PipelineStageFlags::TRANSFER;
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
            image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            image_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            source_stage = vk::PipelineStageFlags::TRANSFER;
            destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        } else if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        {
            image_barrier.src_access_mask = vk::AccessFlags::empty();
            image_barrier.dst_access_mask = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;

            source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
            destination_stage = vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS;
        } else {
            error!("Unsupported layout transition!");
            panic!("Unsupported layout transition!");
        }

        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_barrier],
            );
        }

        Self::end_single_time_commands(device, &command_buffer, command_pool, graphics_queue)
    }

    pub fn find_memorytype_index(
        memory_req: &vk::MemoryRequirements,
        memory_prop: &vk::PhysicalDeviceMemoryProperties,
        flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        memory_prop.memory_types[..memory_prop.memory_type_count as _]
            .iter()
            .enumerate()
            .find(|(index, memory_type)| {
                (1 << index) & memory_req.memory_type_bits != 0
                    && memory_type.property_flags & flags == flags
            })
            .map(|(index, _memory_type)| index as _)
    }

    fn create_render_pass(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        swapchain_image_format: &vk::Format,
    ) -> VkResult<vk::RenderPass> {
        let color_attachment = vk::AttachmentDescription::default()
            .format(*swapchain_image_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        // There is a mentioned of this layout here: https://vkguide.dev/docs/new_chapter_1/vulkan_mainloop_code/
        // the `final_layout` in the article is `VK_IMAGE_LAYOUT_GENERAL` but I'm using `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        // The index of the attachment in this array is directly referenced from the fragment shader with the
        // layout(location = 0) out vec4 outColor directive!
        // The following other types of attachments can be referenced by a subpass:

        //   pInputAttachments: Attachments that are read from a shader
        //   pResolveAttachments: Attachments used for multisampling color attachments
        //   pDepthStencilAttachment: Attachment for depth and stencil data
        //   pPreserveAttachments: Attachments that are not used by this subpass, but for which the data must be preserved

        let depth_attachment = vk::AttachmentDescription::default()
            .format(Self::find_depth_format(instance, physical_device))
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let depth_attachment_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachments_ref = [color_attachment_ref];

        // Unlike color attachments, a subpass can only use a single depth (+stencil) attachment.
        // It wouldn't really make any sense to do depth tests on multiple buffers.
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachments_ref)
            .depth_stencil_attachment(&depth_attachment_ref);

        let at = [color_attachment, depth_attachment];
        let sp = [subpass];
        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );
        // Finally, we need to extend our subpass dependencies to make sure that there is no conflict between the transitioning of the depth image and it being cleared
        // as part of its load operation. The depth image is first accessed in the early fragment test pipeline stage and because we have a load operation that clears,
        // we should specify the access mask for writes.

        let dependencies = &[dependency];

        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&at)
            .subpasses(&sp)
            .dependencies(dependencies);
        let render_pass = unsafe { device.create_render_pass(&render_pass_create_info, None)? };
        Ok(render_pass)
    }

    fn create_shader_module(device: &ash::Device, code: &[u32]) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo::default().code(code);
        unsafe { device.create_shader_module(&create_info, None).unwrap() }
    }

    fn read_shader_from_file<P: AsRef<std::path::Path> + std::fmt::Debug>(path: P) -> Vec<u32> {
        let mut file = match std::fs::File::open(&path) {
            Ok(file) => std::io::BufReader::new(file),
            Err(_) => panic!("Failed to open file: {:?}", &path),
        };
        ash::util::read_spv(&mut file).unwrap()
    }

    fn create_descriptor_set_layout(device: &ash::Device) -> VkResult<vk::DescriptorSetLayout> {
        let descriptor_set_layout_binding = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_set_layout_binding);
        unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None) }
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        swapchain_extent: &vk::Extent2D,
        render_pass: &vk::RenderPass,
        set_layouts: &[vk::DescriptorSetLayout],
    ) -> VkResult<(vk::Pipeline, vk::PipelineLayout)> {
        let vert_shader_code =
            Self::read_shader_from_file(concat!(env!("OUT_DIR"), "/shaders/shader.vert"));
        let frag_shader_code =
            Self::read_shader_from_file(concat!(env!("OUT_DIR"), "/shaders/shader.frag"));

        let vert_shader_module = Self::create_shader_module(device, &vert_shader_code);
        let frag_shader_module = Self::create_shader_module(device, &frag_shader_code);

        let vert_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(c"main");

        let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(c"main");

        let shader_stages = [vert_shader_stage_info, frag_shader_stage_info];

        let vertex_attrs_descriptions = Vertex::get_attribute_descriptions();
        let vertex_binds_descriptions = Vertex::get_binding_description();
        let vert_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&vertex_attrs_descriptions)
            .vertex_binding_descriptions(&vertex_binds_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // let viewport = vk::Viewport::default()
        //     .x(0.0)
        //     .y(0.0)
        //     .width(swapchain_extent.width as f32)
        //     .height(swapchain_extent.height as f32)
        //     .min_depth(0.0)
        //     .max_depth(1.0);
        // let scissor = vk::Rect2D::default()
        //     .offset(vk::Offset2D::default())
        //     .extent(*swapchain_extent);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment_state = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachment_state);

        let dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(set_layouts);

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create pipeline layout")
        };

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vert_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            .render_pass(*render_pass)
            .subpass(0)
            // A depth stencil state must always be specified if the render pass contains a depth stencil attachment.
            .depth_stencil_state(&depth_stencil_state)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .unwrap()
        };

        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        };

        return Ok((*pipeline.first().unwrap(), pipeline_layout));
    }

    fn required_extension_names(entry: &ash::Entry) -> Vec<*const c_char> {
        // let mut extension_names = vec![Surface::name().as_ptr(), DebugUtils::name().as_ptr()];
        let mut extension_names = vec![surface::NAME.as_ptr(), debug_utils::NAME.as_ptr()];

        #[cfg(windows)]
        extension_names.extend([win32_surface::NAME.as_ptr()]);

        #[cfg(target_os = "macos")]
        extension_names.extend([
            metal_surface::NAME.as_ptr(),
            ash::khr::portability_enumeration::NAME.as_ptr(),
            ash::khr::get_physical_device_properties2::NAME.as_ptr(),
        ]);

        #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
        extension_names.extend([
            ash::extensions::khr::XlibSurface::name().as_ptr(),
            ash::extensions::khr::XcbSurface::name().as_ptr(),
            ash::extensions::khr::WaylandSurface::name().as_ptr(),
        ]);

        #[cfg(debug_assertions)]
        {
            let instance_extensions = unsafe {
                entry
                    .enumerate_instance_extension_properties(None)
                    .expect("Failed to enumerate instance extensions")
            };
            debug!("These instance extensions are available to use: ");

            for instance_extension in instance_extensions.iter() {
                debug!(
                    "{:?}",
                    instance_extension.extension_name_as_c_str().unwrap()
                );
            }
            let debug_extension_names: Vec<&str> = extension_names
                .iter()
                .map(|name: &*const c_char| unsafe {
                    CStr::from_ptr(*name)
                        .to_str()
                        .expect("Failed to convert *const c_char back to CStr")
                })
                .collect();

            debug!(
                "The following instance extensions have been enabled: {:#?}",
                debug_extension_names
            );
        }

        extension_names
    }

    fn check_validation_layer_support(entry: &ash::Entry) -> Result<Vec<*const c_char>, String> {
        let layer_properties = unsafe {
            entry
                .enumerate_instance_layer_properties()
                .expect("Failed to enumerate instance layers")
        };

        debug!("Available layers for the instance: ");
        #[cfg(debug_assertions)]
        {
            for layer_prop in layer_properties.iter() {
                let layer_name =
                    unsafe { std::ffi::CStr::from_ptr(layer_prop.layer_name.as_ptr()) };
                debug!("{:?}", layer_name);
            }
        }

        let layer_names = vec![c"VK_LAYER_KHRONOS_validation"];

        for layer_name in layer_names.iter() {
            let mut layer_found = false;

            for layer_prop in layer_properties.iter() {
                let test_name2 =
                    unsafe { std::ffi::CStr::from_ptr(layer_prop.layer_name.as_ptr()) };

                if *layer_name == test_name2 {
                    layer_found = true;
                    break;
                }
            }

            if !layer_found {
                let err_msg = format!(
                    "Validation layer not found: {}",
                    layer_name
                        .to_str()
                        .expect("Couldn't convert layer name to string")
                );
                error!("{:?}", err_msg);
                return Err(err_msg);
            }
        }
        debug!("Validation layers picked: {:?}", layer_names);
        Ok(layer_names.iter().map(|name| name.as_ptr()).collect())
    }

    fn populate_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
        vk::DebugUtilsMessengerCreateInfoEXT::default()
            .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_utils_callback))
    }

    fn setup_debug_messenger(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)> {
        #[cfg(not_debug_assertions)]
        return None;

        let debug_create_info: DebugUtilsMessengerCreateInfoEXT =
            Self::populate_debug_messenger_create_info();

        let debug_utils_loader = debug_utils::Instance::new(entry, instance);
        let debug_messenger = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&debug_create_info, None)
                .expect("Failed to create debug messenger")
        };

        Some((debug_utils_loader, debug_messenger))
    }

    fn draw_frame(&mut self) {
        self.clock.tick();
        self.clock.is_valid = false;
        // println!("{:?}", self.clock);

        unsafe {
            let curr_fence = self.in_flight_fences[self.current_frame];
            let curr_command_buffer = self.command_buffers[self.current_frame];
            let curr_image_available_semaphore =
                self.image_available_semaphores[self.current_frame];
            let curr_render_finished_sempahore =
                self.render_finished_semaphores[self.current_frame];

            self.device
                .wait_for_fences(&[curr_fence], true, u64::MAX)
                .unwrap();

            let result = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                std::u64::MAX,
                curr_image_available_semaphore,
                vk::Fence::null(),
            );

            let image_index = match result {
                Ok((image_index, _)) => image_index,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    info!("Swapchain recreation in progress!");
                    self.recreate_swapchain();
                    return;
                }
                Err(e) if e != vk::Result::SUBOPTIMAL_KHR => {
                    panic!("Failed to acquire next image: {:?}", e);
                }
                Err(_) => {
                    panic!("Failed to acquire next image")
                }
            };

            self.update_uniform_buffer();

            self.device.reset_fences(&[curr_fence]).unwrap();
            self.device
                .reset_command_buffer(curr_command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
            self.record_command_buffer(&curr_command_buffer, image_index as usize);

            let wait_semaphores = [curr_image_available_semaphore];
            let signal_semaphores = [curr_render_finished_sempahore];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [curr_command_buffer];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);

            self.device
                .queue_submit(self.graphics_queue, &[submit_info], curr_fence)
                .expect("Failed to submit queue");

            let swapchains = [self.swapchain];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            let result = self
                .swapchain_loader
                .queue_present(self.present_queue, &present_info);

            match result {
                Ok(_) => {}
                Err(e) => {
                    if e == vk::Result::ERROR_OUT_OF_DATE_KHR
                        || e == vk::Result::SUBOPTIMAL_KHR
                        || self.frame_buffer_resized
                    {
                        info!("Swapchain recreation in progress!");
                        self.recreate_swapchain();
                    } else {
                        panic!("Failed to present swapchain image to the queue: {:?}", e);
                    }
                }
            };

            self.current_frame = (self.current_frame + 1) % Self::MAX_FRAMES_IN_FLIGHT;
        }
        self.clock.is_valid = true;
    }

    fn update_uniform_buffer(&mut self) {
        let elapsed = self.clock.total_elapsed_time.as_secs_f32();
        let mut proj = glam::Mat4::perspective_rh(
            45.0 * PI / 180.0,
            self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32,
            0.1,
            10.0,
        );
        // i.e: proj[1][1] *= -1.0;
        proj.col_mut(1)[1] *= -1.0;

        let ubo = UniformBufferObject {
            // model: Mat4::from_rotation_y(elapsed * (90.0 * PI / 180.0)),
            // model: Mat4::IDENTITY,
            model: Mat4::from_rotation_z(elapsed * (90.0 * PI / 180.0)),
            view: Mat4::look_at_rh(Vec3::new(2.0, 2.0, 2.0), Vec3::new(0.0, 0.0, 0.0), Vec3::Z),
            proj,
        };

        unsafe {
            // TODO revisit this
            let data = self.uniform_buffers_mapped[self.current_frame] as *mut UniformBufferObject;
            *data = ubo;
        }
    }

    fn record_command_buffer(&self, command_buffer: &vk::CommandBuffer, image_index: usize) {
        unsafe {
            self.device
                .begin_command_buffer(*command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();
        };

        // Because we now have multiple attachments with VK_ATTACHMENT_LOAD_OP_CLEAR, we also need to specify multiple clear values.
        // Note that the order of clearValues should be identical to the order of your attachments.
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.2, 0.1, 0.1, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.swapchain_frame_buffers[image_index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_extent,
            })
            .clear_values(&clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                *command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            self.device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            let viewport = vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(self.swapchain_extent.width as f32)
                .height(self.swapchain_extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            self.device
                .cmd_set_viewport(*command_buffer, 0, &[viewport]);

            let scissor = vk::Rect2D::default()
                .offset(vk::Offset2D::default())
                .extent(self.swapchain_extent);
            self.device.cmd_set_scissor(*command_buffer, 0, &[scissor]);

            // new
            let buffers = [self.vertex_buffer];
            let offsets = [0];
            self.device
                .cmd_bind_vertex_buffers(*command_buffer, 0, &buffers, &offsets);

            self.device.cmd_bind_index_buffer(
                *command_buffer,
                self.index_buffer,
                0,
                vk::IndexType::UINT32,
            );

            // replaced by model indices
            // self.device.cmd_bind_index_buffer(
            //     *command_buffer,
            //     self.index_buffer,
            //     0,
            //     vk::IndexType::UINT16,
            // );

            self.device.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[self.current_frame]],
                &[],
            );

            // self.device.cmd_draw(*command_buffer, 3, 1, 0, 0);
            // replaced by model
            // self.device
            //     .cmd_draw_indexed(*command_buffer, Self::INDICES.len() as u32, 1, 0, 0, 0);
            self.device.cmd_draw_indexed(
                *command_buffer,
                self.model_indices.len() as u32,
                1,
                0,
                0,
                0,
            );

            self.device.cmd_end_render_pass(*command_buffer);

            self.device
                .end_command_buffer(*command_buffer)
                .expect("Failed to record command buffer");
        }
    }

    fn create_sync_objects(
        device: &ash::Device,
    ) -> VkResult<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>)> {
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let max_frames_in_flight = Self::MAX_FRAMES_IN_FLIGHT;
        let mut image_available_semaphores = Vec::with_capacity(max_frames_in_flight);
        let mut render_finished_semaphores = Vec::with_capacity(max_frames_in_flight);
        let mut in_flight_fences = Vec::with_capacity(max_frames_in_flight);

        for _ in 0..Self::MAX_FRAMES_IN_FLIGHT {
            unsafe {
                let image_available_semaphore =
                    device.create_semaphore(&semaphore_create_info, None)?;
                let render_finished_semaphore =
                    device.create_semaphore(&semaphore_create_info, None)?;
                let in_flight_fence = device.create_fence(&fence_create_info, None)?;

                image_available_semaphores.push(image_available_semaphore);
                render_finished_semaphores.push(render_finished_semaphore);
                in_flight_fences.push(in_flight_fence);
            }
        }
        Ok((
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        ))
    }

    fn create_command_buffers(
        device: &ash::Device,
        command_pool: &vk::CommandPool,
    ) -> VkResult<Vec<vk::CommandBuffer>> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(Self::MAX_FRAMES_IN_FLIGHT as u32);

        unsafe { device.allocate_command_buffers(&allocate_info) }
    }

    fn create_command_pool(
        device: &ash::Device,
        queue_family_indices: &QueueFamilyIndices,
    ) -> VkResult<vk::CommandPool> {
        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_indices.graphics_family.unwrap())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        unsafe { device.create_command_pool(&pool_create_info, None) }
    }

    fn create_frame_buffers(
        device: &ash::Device,
        swapchain_images_views: &[vk::ImageView],
        render_pass: &vk::RenderPass,
        swapchain_extent: &vk::Extent2D,
        depth_image_view: vk::ImageView,
    ) -> VkResult<Vec<vk::Framebuffer>> {
        let mut frame_buffers = Vec::with_capacity(swapchain_images_views.len());
        debug!(
            "swapchain_images_views len: {}",
            swapchain_images_views.len()
        );

        for &image_view in swapchain_images_views.iter() {
            let attachments = [image_view, depth_image_view];

            let frame_buffer_create_info = vk::FramebufferCreateInfo::default()
                .render_pass(*render_pass)
                .attachments(&attachments)
                .width(swapchain_extent.width)
                .height(swapchain_extent.height)
                .layers(1);

            match unsafe { device.create_framebuffer(&frame_buffer_create_info, None) } {
                Ok(frame_buffer) => frame_buffers.push(frame_buffer),
                Err(err) => return Err(err),
            }
        }

        Ok(frame_buffers)
    }

    fn recreate_swapchain(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            // TODO handle minimization case...
            warn!("Swapchain recreation started!");
            self.cleanup_swapchain();

            (
                self.swapchain,
                self.swapchain_loader,
                self.swapchain_images,
                self.swapchain_image_format,
                self.swapchain_extent,
            ) = Self::create_swapchain(
                &self.instance,
                &self.device,
                &self.surface,
                &self.swapchain_support_details,
                self.window.inner_size().into(),
                &self.queue_family_indices,
            )
            .unwrap();

            self.swapchain_images_views = Self::create_swapchain_images_views(
                &self.device,
                &self.swapchain_images,
                self.swapchain_image_format,
            )
            .unwrap();

            (
                self.depth_image,
                self.depth_image_memory,
                self.depth_image_view,
            ) = Self::create_depth_resources(
                &self.device,
                &self.instance,
                &self.physical_device,
                &self.command_pool,
                &self.graphics_queue,
                &self.swapchain_extent,
                &self.device_memory_properties,
            )
            .unwrap();

            self.swapchain_frame_buffers = Self::create_frame_buffers(
                &self.device,
                &self.swapchain_images_views,
                &self.render_pass,
                &self.swapchain_extent,
                self.depth_image_view,
            )
            .unwrap();
        }
    }

    fn cleanup_swapchain(&self) {
        unsafe {
            for i in 0..self.swapchain_frame_buffers.len() {
                self.device
                    .destroy_framebuffer(self.swapchain_frame_buffers[i], None);
            }

            for i in 0..self.swapchain_images_views.len() {
                self.device
                    .destroy_image_view(self.swapchain_images_views[i], None);
            }

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_image_memory, None);

            self.device.destroy_sampler(self.texture_sampler, None);

            self.device
                .destroy_image_view(self.texture_image_view, None);
            self.device.destroy_image(self.texture_image, None);
            self.device.free_memory(self.texture_image_memory, None);

            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);

            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);

            self.cleanup_swapchain();

            for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
                self.device.destroy_buffer(self.uniform_buffers[i], None);
                self.device
                    .free_memory(self.uniform_buffers_memory[i], None);
            }

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            self.device.destroy_pipeline(self.pipeline, None);

            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            self.device.destroy_render_pass(self.render_pass, None);

            for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }

            self.device.destroy_command_pool(self.command_pool, None);

            self.device.destroy_device(None);

            if let Some((debug_utils, debug_messenger)) = self.debug_messenger.take() {
                debug_utils.destroy_debug_utils_messenger(debug_messenger, None);
            }

            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_file(true)
        .with_line_number(true)
        .with_target(false)
        .with_max_level(tracing::Level::TRACE)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    #[cfg(debug_assertions)]
    {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("shaders")
            .join("shader.frag");
        debug!("path of shaders: {:?}", path); // appears to be right
    }

    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event()
        .build()
        .unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Vulkan Port")
            .build(&event_loop)
            .unwrap(),
    );

    let mut app = match VulkanApp::new(window.clone()) {
        Ok(app) => {
            info!("VulkanApp created successfully");
            app
        }
        Err(_) => {
            std::process::exit(0);
        }
    };

    event_loop.set_control_flow(ControlFlow::Poll);

    let event_handler = move |event, event_loop: &EventLoopWindowTarget<UserEvent>| {
        handle_winit_event(&mut app, event, event_loop);
    };

    if let Err(err) = event_loop.run(event_handler) {
        error!("{:?}", err);
        panic!();
    }
}

fn handle_winit_event(
    app: &mut VulkanApp,
    event: Event<UserEvent>,
    event_loop: &EventLoopWindowTarget<UserEvent>,
) {
    match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        }
        | Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::KeyQ),
                            ..
                        },
                    ..
                },
            ..
        } => {
            println!("The close button was pressed; stopping");
            event_loop.exit();
        }
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => {
            app.draw_frame();
            app.window.request_redraw();
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(physical_size),
            ..
        } => {
            app.frame_buffer_resized = true;
        }
        _ => (),
    }
}
