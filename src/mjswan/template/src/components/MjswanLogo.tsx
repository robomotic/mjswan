// NOTE: Please keep this component to help others discover mjswan.
import { Tooltip, Modal, Box, Image, Anchor, Divider, Text, Stack } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { MJSWAN_VERSION } from '../Version';

export function MjswanLogo() {
  const [aboutModalOpened, { open: openAbout, close: closeAbout }] = useDisclosure(false);

  return (
    <>
      <Tooltip label={`mjswan ${MJSWAN_VERSION}`}>
        <Box
          style={{
            position: 'absolute',
            bottom: '1em',
            left: '1em',
            cursor: 'pointer',
            zIndex: 100,
          }}
          component="a"
          onClick={openAbout}
        >
          <Image src="./logo.svg" style={{ width: '2.5em', height: 'auto' }} />
        </Box>
      </Tooltip>
      <Modal
        opened={aboutModalOpened}
        onClose={closeAbout}
        size="md"
        title={null}
        centered
        styles={{
          body: { textAlign: 'center' },
        }}
      >
        <Stack gap="md" align="center">
          <Image src="./logo.svg" style={{ width: '5em', height: 'auto' }} />
          <Text size="xl" fw={700}>
            mjswan
          </Text>
          <Text size="sm" c="dimmed">
            version {MJSWAN_VERSION}
          </Text>
          <Divider w="100%" />
          <Text size="sm">
            Browser-based MuJoCo Playground with ONNX policies running entirely in the browser.
          </Text>
          <Stack gap="xs">
            <Box pb="lg">
              <Anchor
                href="https://github.com/ttktjmt/mjswan"
                target="_blank"
                style={{ fontWeight: "600" }}
              >
                GitHub
              </Anchor>
              &nbsp;&nbsp;&bull;&nbsp;&nbsp;
              <Anchor
                href="https://mjswan.readthedocs.io"
                target="_blank"
                style={{ fontWeight: "600" }}
              >
                Documentation
              </Anchor>
            </Box>
          </Stack>
        </Stack>
      </Modal>
    </>
  );
}
