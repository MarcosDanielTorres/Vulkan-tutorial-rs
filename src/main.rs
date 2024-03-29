use ash::extensions::ext::debug_utils;
use ash::extensions::mvk::macos_surface;
use ash::extensions::khr::win32_surface;
use ash::vk::ext::queue_family_foreign;
use ash::vk::{PFN_vkEnumerateInstanceExtensionProperties, PresentModeKHR, SurfaceCapabilitiesKHR, SurfaceFormatKHR, SurfaceKHR};
use ash::vk::{self, DebugUtilsMessengerCreateInfoEXT};
use tracing_subscriber::EnvFilter;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::ffi::{c_char, CStr};
use std::fmt;
use std::io::Write;
use std::{any::Any, sync::Arc};
use tracing::{debug, error, info, trace, warn};
use winit::event_loop::EventLoopBuilder;
use winit::raw_window_handle::{ HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use winit::{
    error::EventLoopError,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    window::Window,
};

use ash::extensions::khr::surface;

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


const DEVICE_ENABLED_EXTENSION_NAMES: [*const c_char; 1] = [ash::extensions::khr::swapchain::NAME.as_ptr()];
struct VulkanApp {
    // window
    window: Arc<Window>,

    instance: ash::Instance,

    // devices
    physical_device: vk::PhysicalDevice,
    device: ash::Device,

    // queue
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    // surface
    surface_loader: surface::Instance,
    surface: SurfaceKHR,

    // debug
    debug_messenger: Option<(ash::extensions::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl VulkanApp {
    pub unsafe fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) -> ash::prelude::VkResult<vk::SurfaceKHR> {
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
            },
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

        let enabled_extension_names = &Self::required_extension_names(&entry);

        // this is a bug, its being used...
        #[allow(unused_assignments)]
        let mut enabled_layer_names = vec![];

        #[allow(unused_assignments)]
        let mut debug_change_name: Option<DebugUtilsMessengerCreateInfoEXT> = None;

        #[cfg(debug_assertions)]
        {
            match Self::check_validation_layer_support(&entry) {
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
            .enabled_extension_names(&enabled_extension_names)
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

    fn create_physical_device(instance: &ash::Instance, surface_loader: &surface::Instance, surface: &SurfaceKHR) -> Result<(vk::PhysicalDevice, QueueFamilyIndices), String> {
        unsafe {
            let physical_devices = instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices");

            if physical_devices.is_empty() {
                let err_msg = "No GPU with Vulkan support found!";
                error!(err_msg);
                return Err(err_msg.to_string());
            }

            let device_enabled_extension_names_as_c_strs = DEVICE_ENABLED_EXTENSION_NAMES.iter().map(|&x| CStr::from_ptr(x)).collect::<Vec<_>>();

            for physical_device in physical_devices.iter() {
                let properties = instance.get_physical_device_properties(*physical_device);
                let features = instance.get_physical_device_features(*physical_device);
                let device_name = CStr::from_ptr(properties.device_name.as_ptr())
                    .to_str()
                    .expect("Failed to get device name");
                let device_type = properties.device_type;

                let mut required_extensions = HashSet::new();
                required_extensions.extend(device_enabled_extension_names_as_c_strs.clone());

                let available_extensions = instance.enumerate_device_extension_properties(*physical_device).unwrap();

                for extension in available_extensions.iter() {
                    debug!("{:?}", &extension.extension_name_as_c_str().unwrap());
                    required_extensions.remove(&extension.extension_name_as_c_str().unwrap());
                }

                // this is `isDeviceSuitable` in C++
                if device_type == vk::PhysicalDeviceType::DISCRETE_GPU && required_extensions.is_empty()
                {
                    info!(
                        "Physical Device (GPU): {}, Device Type: {:?}",
                        device_name, device_type
                    );
                    let queue_families_indices = Self::find_queue_families(&instance, &physical_device, &surface_loader, &surface);
                    return Ok((*physical_device, queue_families_indices));
                }
            }
        }

        let err_msg = "No suitable GPU found!".to_string();
        error!("{:?}", err_msg);
        Err(err_msg)
    }

    
    fn find_queue_families(instance: &ash::Instance, physical_device: &vk::PhysicalDevice, surface_loader: &surface::Instance, surface: &SurfaceKHR) -> QueueFamilyIndices {
        unsafe {
            let queue_families = instance.get_physical_device_queue_family_properties(*physical_device);
            let mut indices = QueueFamilyIndices {
                graphics_family: None,
                present_family: None,
            };

            for (i, queue_family) in queue_families.iter().enumerate() {
                if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    indices.graphics_family = Some(i as u32);
                }

                let is_present_support = surface_loader.get_physical_device_surface_support(*physical_device, i as u32, *surface).unwrap();

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

    fn create_logical_device(instance: &ash::Instance, physical_device: &vk::PhysicalDevice, indices: QueueFamilyIndices) -> Result<ash::Device, String> {
        let mut unique_queue_families = HashSet::new();
        unique_queue_families.insert(indices.graphics_family.unwrap()); 
        unique_queue_families.insert(indices.present_family.unwrap());

        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo<'static>> = unique_queue_families
            .iter()
            .map(|queue_family_index| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*queue_family_index)
                    .queue_priorities(&[1.0])
            }).collect();

        let device_features = vk::PhysicalDeviceFeatures::default();


        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&DEVICE_ENABLED_EXTENSION_NAMES)
            .enabled_features(&device_features);

        unsafe {
            match instance.create_device(*physical_device, &create_info, None) {
                Ok(device) => Ok(device),
                Err(err) => {
                    let err_msg = format!("Failed to create logical device: {:?}", err);
                    error!("{:?}", err_msg);
                    return Err(err_msg)
                },
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

        let surface = unsafe { Self::create_surface(&entry, &instance, display_handle, window_handle, None).unwrap() } ;
        let surface_loader = surface::Instance::new(&entry, &instance);

        let (physical_device, queue_family_indices) = Self::create_physical_device(&instance, &surface_loader, &surface)?;
        let device = Self::create_logical_device(&instance, &physical_device, queue_family_indices)?;
        let graphics_queue = unsafe { device.get_device_queue(queue_family_indices.graphics_family.unwrap(), 0) };
        let present_queue = unsafe { device.get_device_queue(queue_family_indices.present_family.unwrap(), 0) };



        unsafe {
            let surface_format = surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()[0];

            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap();
        }

        //let surface_khr = ;

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
        })
    }

    fn required_extension_names(entry: &ash::Entry) -> Vec<*const c_char> {
        // let mut extension_names = vec![Surface::name().as_ptr(), DebugUtils::name().as_ptr()];
        let mut extension_names = vec![ash::extensions::khr::surface::NAME.as_ptr(), ash::extensions::ext::debug_utils::NAME.as_ptr()];

        #[cfg(all(windows))]
        extension_names.extend([ash::extensions::khr::win32_surface::NAME.as_ptr()]);

        #[cfg(target_os = "macos")]
        extension_names.extend([
            MacOSSurface::name().as_ptr(),
            //vk::KhrPortabilityEnumerationFn::name().as_ptr(),
            vk::khr::portability_enumeration::NAME.as_ptr(),
            //vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
            vk::khr::get_physical_device_properties2::NAME.as_ptr(),
        ]);

        #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
        extension_names.extend([
            ash::extensions::khr::XlibSurface::name().as_ptr(),
            ash::extensions::khr::XcbSurface::name().as_ptr(),
            ash::extensions::khr::WaylandSurface::name().as_ptr(),
        ]);

        #[cfg(debug_assertions)]
        {
            let instance_extensions = unsafe { entry
                .enumerate_instance_extension_properties(None)
                .expect("Failed to enumerate instance extensions")};
            debug!(
                "These instance extensions are available to use: {:#?}",
                instance_extensions
            );

            let debug_extension_names: Vec<&str> = extension_names
                .iter()
                .map(|name: &*const c_char| unsafe {
                    CStr::from_ptr(name.clone())
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
        let layer_properties = unsafe { entry
            .enumerate_instance_layer_properties()
            .expect("Failed to enumerate instance layers") };


        let layer_names: Vec<&str> = vec!["VK_LAYER_KHRONOS_validation"];

        for layer_name in layer_names.iter() {
            let mut layer_found = false;

            for layer_prop in layer_properties.iter() {
                let test_name2 = unsafe {
                    std::ffi::CStr::from_ptr(layer_prop.layer_name.as_ptr())
                        .to_str()
                        .expect("Failed to get layer name")
                };

                if *layer_name == test_name2 {
                    layer_found = true;
                    break;
                }
            }

            if !layer_found {
                let err_msg = format!("Validation layer not found: {}", layer_name);
                error!("{:?}", err_msg);
                return Err(err_msg);
            }
        }
        debug!("Validation layers found: {:?}", layer_names);
        Ok(layer_names
            .iter()
            .map(|name| name.as_ptr() as *const i8)
            .collect())
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
    ) -> Option<(ash::extensions::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)> {
        #[cfg(not_debug_assertions)]
        return None;

        let debug_create_info: DebugUtilsMessengerCreateInfoEXT =
            Self::populate_debug_messenger_create_info();

        let debug_utils_loader = ash::extensions::ext::debug_utils::Instance::new(entry, instance);
        let debug_messenger = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&debug_create_info, None)
                .expect("Failed to create debug messenger")
        };

        Some((debug_utils_loader, debug_messenger))
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);

            self.device.destroy_device(None);

            if let Some((debug_utils, debug_messenger)) = self.debug_messenger.take() {
                debug_utils.destroy_debug_utils_messenger(debug_messenger, None);
            }

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
        } => {
            println!("The close button was pressed; stopping");
            event_loop.exit();
        }
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => {}
        _ => (),
    }
}
