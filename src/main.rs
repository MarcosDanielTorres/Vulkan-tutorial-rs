use ash::ext::debug_utils;
use ash::ext::metal_surface;
use ash::khr::{swapchain, win32_surface};
use ash::prelude::VkResult;
use ash::vk::{
    self, DebugUtilsMessengerCreateInfoEXT, PhysicalDeviceFeatures2KHR,
    PhysicalDeviceVulkan13Features, PresentModeKHR, SurfaceCapabilitiesKHR, SurfaceFormatKHR,
    SurfaceKHR,
};
use std::collections::HashSet;
use std::ffi::{c_char, CStr};
use std::fmt;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;
use winit::{
    event::{Event, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder, EventLoopWindowTarget},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle},
    window::Window,
};

// use ash::extensions::khr::surface;
use ash::khr::surface;

#[derive(Copy, Clone)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 3],
}

impl Vertex {
    fn get_binding_description() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)]
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::size_of::<[f32; 2]>() as u32),
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

enum BufferType {
    Vertex,
    Index,
}

struct VulkanApp {
    // window
    window: Arc<Window>,

    instance: ash::Instance,

    // devices
    physical_device: vk::PhysicalDevice,
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
    pipeline_layout: vk::PipelineLayout,

    // buffers
    // these two go together, probably should have them inside a struct and deallocated them together
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,

    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    // command buffer
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    // sync objects
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,

    // debug
    debug_messenger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl VulkanApp {
    const MAX_FRAMES_IN_FLIGHT: usize = 2;

    const VERTICES: [Vertex; 4] = [
        Vertex {
            pos: [-0.5, -0.5],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            pos: [0.5, -0.5],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            pos: [0.5, 0.5],
            color: [0.0, 0.0, 1.0],
        },
        Vertex {
            pos: [-0.5, 0.5],
            color: [1.0, 1.0, 1.0],
        },
    ];

    const INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];
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
        let surface_format: SurfaceFormatKHR = *swapchain_support_details
            .formats
            .iter()
            .find(|format| {
                // TODO Why using B8G8R8A8_SRGB does not merge colors as B8G8R8A8_UNORM does
                // format.format == vk::Format::B8G8R8A8_SRGB
                format.format == vk::Format::B8G8R8A8_UNORM
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
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
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
                                surface_format.format,
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
            let create_info = vk::ImageViewCreateInfo::default()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(swapchain_image_format)
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
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            match unsafe { device.create_image_view(&create_info, None) } {
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

        let render_pass = Self::create_render_pass(&device, &swapchain_image_format).unwrap();

        Self::create_descriptor_set_layout();

        let (pipeline, pipeline_layout) =
            Self::create_graphics_pipeline(&device, &swapchain_extent, &render_pass).unwrap();

        let swapchain_frame_buffers = Self::create_frame_buffers(
            &device,
            &swapchain_images_views,
            &render_pass,
            &swapchain_extent,
        )
        .unwrap();

        let command_pool = Self::create_command_pool(&device, &queue_family_indices).unwrap();

        let (vertex_buffer, vertex_buffer_memory) = Self::create_buffers(
            &device,
            &command_pool,
            &graphics_queue,
            &device_memory_properties,
            Self::VERTICES.as_ptr(),
            BufferType::Vertex,
            std::mem::size_of_val(&Self::VERTICES) as u64,
        )
        .unwrap();

        let (index_buffer, index_buffer_memory) = Self::create_buffers(
            &device,
            &command_pool,
            &graphics_queue,
            &device_memory_properties,
            Self::INDICES.as_ptr(),
            BufferType::Index,
            std::mem::size_of_val(&Self::INDICES) as u64,
        )
        .unwrap();

        let command_buffers = Self::create_command_buffers(&device, &command_pool).unwrap();

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            Self::create_sync_objects(&device).unwrap();

        Ok(Self {
            window,

            instance,
            debug_messenger,
            physical_device,
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

            pipeline_layout,
            pipeline,

            vertex_buffer,
            vertex_buffer_memory,

            index_buffer,
            index_buffer_memory,

            command_pool,
            command_buffers,

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
                // TODO Do error handling
                unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0) }.unwrap();
                buffer_memory
            }
            Err(err) => {
                let err_msg = format!("Failed to allocate vertex buffer memory: {:?}", err);
                error!("{:?}", err_msg);
                panic!("{:?}", err_msg);
            }
        };

        Ok((buffer, buffer_memory))
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

            let regions = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(size);
            device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &[regions]);

            device
                .end_command_buffer(command_buffer)
                .expect("Failed to record command buffer");

            let command_buffers = [command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device
                .queue_submit(*graphics_queue, &[submit_info], vk::Fence::null())
                .expect("Failed to submit queue");

            device.queue_wait_idle(*graphics_queue).unwrap();
            device.free_command_buffers(*command_pool, &command_buffers)
        };
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
        let binding = [color_attachment_ref];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&binding);

        let at = [color_attachment];
        let sp = [subpass];
        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            );

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

    fn read_shader_from_file<P: AsRef<std::path::Path>>(path: P) -> Vec<u32> {
        let mut file = std::fs::File::open(path).unwrap();
        ash::util::read_spv(&mut file).unwrap()
    }

    fn create_descriptor_set_layout() {
       let descriptor_set_layout_binding = vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1);
       
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        swapchain_extent: &vk::Extent2D,
        render_pass: &vk::RenderPass,
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
            .cull_mode(vk::CullModeFlags::BACK)
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

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create pipeline layout")
        };

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
            self.device.reset_fences(&[curr_fence]).unwrap();

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
    }

    fn record_command_buffer(&self, command_buffer: &vk::CommandBuffer, image_index: usize) {
        unsafe {
            self.device
                .begin_command_buffer(*command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();
        };

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

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
                vk::IndexType::UINT16,
            );

            // self.device.cmd_draw(*command_buffer, 3, 1, 0, 0);
            self.device
                .cmd_draw_indexed(*command_buffer, Self::INDICES.len() as u32, 1, 0, 0, 0);

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
    ) -> VkResult<Vec<vk::Framebuffer>> {
        let mut frame_buffers = Vec::with_capacity(swapchain_images_views.len());
        debug!(
            "swapchain_images_views len: {}",
            swapchain_images_views.len()
        );

        for &image_view in swapchain_images_views.iter() {
            let attachments = [image_view];

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

            Self::create_swapchain_images_views(
                &self.device,
                &self.swapchain_images,
                self.swapchain_image_format,
            )
            .unwrap();

            Self::create_frame_buffers(
                &self.device,
                &self.swapchain_images_views,
                &self.render_pass,
                &self.swapchain_extent,
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

            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);

            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);

            self.cleanup_swapchain();

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
    let window = Arc::new(Window::new(&event_loop).unwrap());

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
